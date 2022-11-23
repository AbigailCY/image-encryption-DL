import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from ComplexNN import *


class preEncoder(nn.Module):
    def __init__(self):
        super(preEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, I):
        return self.features(I)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(384,192,3,1),
            nn.LeakyReLU(),
            nn.Conv2d(192,64,2,1),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(64, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv4 = ComplexConv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = ComplexConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.Crelu = ComplexRelu()
        self.Cmax = ComplexMaxPool2d(2, 2)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )

    def forward(self, a, b, theta):
        x1, x2 = encode(a, b, theta)
        x1, x2 = self.conv4(x1, x2)
        x1, x2 = self.Crelu(x1, x2)
        x1, x2 = self.Cmax(x1, x2)
        x1, x2 = self.conv5(x1, x2)
        x1, x2 = self.Crelu(x1, x2)
        x1, x2 = self.Cmax(x1, x2)

        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)

        x = decode(x1, x2, theta)
        x = self.classifier(x)

        return x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(trainloader, net, g, discriminator, criterion, optimizer, optimizer_g, optimizer_G, optimizer_D, epoch, device, CLIP_VALUE, CRITIC, K_SYNTHETIC, theta):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    net.train()

    end = time.time()
    for i, (inputs, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, target = inputs.to(device), target.to(device)

        b = g(torch.rand_like(inputs)).detach()
        a = g(inputs).detach()
        # bs = a.shape[0]


        for j in range(CRITIC):
            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = True
            delta_theta = torch.Tensor(K_SYNTHETIC).uniform_(0,np.pi).to(device)
            fake_imgs = torch.stack([discriminator(ED(a, b, dt)) for dt in delta_theta])
            loss_D = -torch.mean(discriminator(a)) + torch.mean(fake_imgs)
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            for p in discriminator.parameters():  # reset requires_grad
                p.requires_grad = False

        # if i % CRITIC == 0:
        optimizer_G.zero_grad()
        a = g(inputs)
        delta_theta = torch.Tensor(1).uniform_(0,np.pi).to(device)
        fake_imgs = ED(a, b, delta_theta)
        loss_g = torch.mean(discriminator(a)) - torch.mean(discriminator(fake_imgs))
        loss_g.backward()
        optimizer_G.step()

        optimizer.zero_grad()
        optimizer_g.zero_grad()
        a = g(inputs)
        output = net(a, b, theta) #[128, 384, 4, 4]
        loss = criterion(output, target)
        loss.backward()
        optimizer_g.step()
        optimizer.step()

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec.item(), inputs.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0: #print frequency
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}), {loss_g:.4f}\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(trainloader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss_g=loss_g*100, top1=top1))
        # if i % 20 == 0: #print frequency
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
        #            epoch, i, len(trainloader), batch_time=batch_time,
        #            data_time=data_time, loss=losses, top1=top1))

def validate(val_loader, g, net, criterion, device):
    theta = torch.Tensor(1).uniform_(0,2*np.pi).to(device) 
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    # net.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)

            # compute output
            b = torch.rand_like(input).to(device)
            b = g(b)
            a = g(input)
            output = net(a, b, theta)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))
    return top1.avg


def main():
    BATCH_SIZE = 128
    LR = 3e-4
    EPOCHS = 150
    CRITIC = 5              # number of training steps for discriminator per iter
    CLIP_VALUE = 0.01       # lower and upper clip value for disc. weights
    K_SYNTHETIC = 5

    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(
        root='./',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)
    g = preEncoder().to(device)
    discriminator = Discriminator().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=LR)
    optimizer_g = optim.RMSprop(g.parameters(), lr=LR)
    optimizer_G = optim.RMSprop(g.parameters(), lr=0.00005)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.00005)

    lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120], gamma=0.2, last_epoch=-1)
    lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, [60, 120], gamma=0.2, last_epoch=-1)
    lr_scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, [60, 120], gamma=0.2, last_epoch=-1)
    lr_scheduler4 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, [60, 120], gamma=0.2, last_epoch=-1)

    theta = torch.Tensor(1).uniform_(0,2*np.pi).to(device) 
    for epoch in range(EPOCHS):
        train(trainloader, net, g, discriminator, criterion, optimizer, optimizer_g, optimizer_G, optimizer_D, epoch, device, CLIP_VALUE, CRITIC, K_SYNTHETIC, theta)
        lr_scheduler1.step()
        lr_scheduler2.step()
        lr_scheduler3.step()
        lr_scheduler4.step()

    print('Finished Training')
    torch.save(net.state_dict(), './check/Alexnet_cifar10_CNN.pth')
    torch.save(g.state_dict(), './check/Alexnet_CNN_g.pkl')
    torch.save(discriminator.state_dict(), './check/Alexnet_cifar10_CNN_D.pth')

    # g.load_state_dict(torch.load('./check/Alexnet_CNN_g.pkl'))
    # net.load_state_dict(torch.load('./check/Alexnet_CNN.pkl'))
    validate(testloader, g, net, criterion, device)

    
if __name__ == '__main__':
    main()
