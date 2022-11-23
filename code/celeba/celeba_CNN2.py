import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import torch.backends.cudnn as cudnn
from ComplexNN import *
from utils import *

parser = argparse.ArgumentParser(description='CelebA Alexnet')
parser.add_argument('--dataroot', type=str,
                    default='/Extra/dataset/face/CelebA/Img', help="celeba dataset path.")
parser.add_argument('--datasets', type=str, default="celeba", help="dataset")
parser.add_argument('--batch_size', type=int, default=128,
                    help="Every train dataset size.")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="starting lr, every 10 epoch decay 10.")
parser.add_argument('--epochs', type=int, default=201, help="Train loop")
parser.add_argument('--phase', type=str, default='eval',
                    help="train or eval? Default:`eval`")
parser.add_argument('--net_path', type=str,
                    default="./check/celeba_alexnet/celeba_epoch_1.pth", help="load model path.")
parser.add_argument('--g_path', type=str,
                    default="./check/celeba_alexnet/celeba_epoch_1.pth", help="load model path.")
# ./check/celeba_alexnet/celeba_epoch_3.pth
opt = parser.parse_args()
print(opt)
CRITIC = 50              # number of training steps for discriminator per iter
CLIP_VALUE = 0.01       # lower and upper clip value for disc. weights
K_SYNTHETIC = 5
try:
    os.makedirs("./check/celeba_alexnet")
except OSError:
    pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True
class preEncoder(nn.Module):
    def __init__(self):
        super(preEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, I):
        return self.features(I)
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(384,192,3,1),
            nn.LeakyReLU(),
            nn.Conv2d(192,64,3,1),
            nn.LeakyReLU(),
            nn.Conv2d(64,32,3,1),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(256,1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.main(x) #[128, 32, 4, 2]
        x = x.view(x.shape[0], -1)  #[128, 256]
        x = self.linear(x)
        x = self.relu(x) #[128, 256]

        return x

class Net(nn.Module):
    def __init__(self, identities=10177):
        super(Net, self).__init__()
        self.conv4 = ComplexConv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = ComplexConv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.Crelu = ComplexRelu()
        self.Cmax = ComplexMaxPool2d(3, 2)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, identities),
        )

    def forward(self, a, b, theta):
        x1, x2 = encode(a, b, theta)
        x1, x2 = self.conv4(x1, x2)
        x1, x2 = self.Crelu(x1, x2)
        x1, x2 = self.conv5(x1, x2)
        x1, x2 = self.Crelu(x1, x2)
        x1, x2 = self.Cmax(x1, x2)

        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)

        x = decode(x1, x2, theta)
        x = self.classifier(x)

        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# 182599
trainset = CelebA(root=opt.dataroot, split='train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                          shuffle=True, num_workers=2)
# 20000
testset = CelebA(root=opt.dataroot, split='test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=2)

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
g = preEncoder().to(device)
discriminator = Discriminator().to(device)
# model.load_state_dict(torch.load(
#     opt.model_path, map_location=lambda storage, loc: storage))
# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
optimizer_g = torch.optim.Adam(g.parameters(), lr=opt.lr)
optimizer_G = torch.optim.RMSprop(g.parameters(), lr=0.00005)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.00005)

lr_scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 120], gamma=0.2, last_epoch=-1)
lr_scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, [60, 120], gamma=0.2, last_epoch=-1)
lr_scheduler3 = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, [60, 120], gamma=0.2, last_epoch=-1)
lr_scheduler4 = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, [60, 120], gamma=0.2, last_epoch=-1)

def train(train_dataloader, net, g, discriminator, criterion, optimizer, optimizer_g, optimizer_G, optimizer_D, epoch, device, CLIP_VALUE, CRITIC, K_SYNTHETIC, theta):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    for i, data in enumerate(train_dataloader):

        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs; data is a list of [inputs, labels]
        inputs, target = data
        inputs = inputs.to(device)
        target = target.to(device)-1
        b = g(torch.rand_like(inputs)).detach()
        a = g(inputs).detach()

        # loss_g = 0.

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

        if i % CRITIC == 0:
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
        output = net(a, b, theta) 
        loss = criterion(output, target)
        loss.backward()
        optimizer_g.step()
        optimizer.step()
        # compute output
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            print(f"Epoch [{epoch + 1}] [{i}/{len(train_dataloader)}]\t"
                  f"Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  f"Loss {loss.item():.4f}\t"
                  f"Loss_g {loss_g:.4f}\t"
                  # f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
                  f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t", end='\r')
    if epoch % 10 == 0:
        torch.save(net.state_dict(),
               f"./check/celeba_alexnet/{opt.datasets}_net_epoch_{epoch + 1}.pth")
        torch.save(g.state_dict(),
               f"./check/celeba_alexnet/{opt.datasets}_g_epoch_{epoch + 1}.pth")
        torch.save(discriminator.state_dict(),
               f"./check/celeba_alexnet/{opt.datasets}_D_epoch_{epoch + 1}.pth")

def valid(dataloader, g, net, criterion, device):
    dim_feature = 10177
    num_valid_sample = 20000
    feature_map = torch.zeros(num_valid_sample, dim_feature).cuda()
    ground_truth = torch.zeros(num_valid_sample).cuda()
    index_begin = 0

    theta = torch.Tensor(1).uniform_(0,2*np.pi).to(device) 
    
    g.eval()
    net.eval()
    with torch.no_grad():
        for step, (X, y) in enumerate(dataloader):
            X, y = X.cuda(), y.cuda()-1
            N = X.size(0)

            b = torch.rand_like(X).to(device)
            b = g(b)
            a = g(X)
            ft = net(a, b, theta)

            index_end = index_begin + N
            feature_map[index_begin:index_end, :] = ft.view(N, dim_feature)
            ground_truth[index_begin:index_end] = y
            index_begin = index_end

        fm_n = feature_map.norm(p=2, dim=1)
        dist = 1 - torch.matmul(feature_map/fm_n.view(num_valid_sample, 1), (feature_map/fm_n.view(num_valid_sample, 1)).t())

        # metrics
        acc = eer(ground_truth, dist)
    return acc

def test(g, model, dataloader):
    # switch to evaluate mode
    model.eval()
    g.eval()
    # init value
    # total = 0.
    # correct = 0.
    top1 = AverageMeter()
    theta = torch.Tensor(1).uniform_(0,2*np.pi).to(device) 
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)-1
            b = torch.rand_like(inputs).to(device)
            b = g(b)
            a = g(inputs)
            outputs = model(a, b, theta)
            prec1 = accuracy(outputs, targets, topk=(1,))
            top1.update(prec1[0], inputs.shape[0])

            # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += (predicted == targets).sum().item()

    # accuracy = 100 * correct / total
    return top1.avg

def run():
    best_prec1 = 0.
    theta = torch.Tensor(1).uniform_(0,2*np.pi).to(device) 
    for epoch in range(opt.epochs):
        # train for one epoch
        print(f"\nBegin Training Epoch {epoch + 1}")
        train(trainloader, net, g, discriminator, criterion, optimizer, optimizer_g, optimizer_G, optimizer_D, epoch, device, CLIP_VALUE, CRITIC, K_SYNTHETIC, theta)
        lr_scheduler1.step()
        lr_scheduler2.step()
        lr_scheduler3.step()
        lr_scheduler4.step()

        # evaluate on test set
        print(f"Begin Validation @ Epoch {epoch + 1}")
        prec1 = test(g, net, testloader)
        # prec1 = valid(testloader, g, net, criterion, device)

        # remember best prec@1 and save checkpoint if desired
        best_prec1 = max(prec1, best_prec1)

        print("Epoch Summary: ")
        print(f"\tEpoch Accuracy: {prec1}")
        print(f"\tBest Accuracy: {best_prec1}")


if __name__ == '__main__':
    if opt.phase == "train":
        run()
    elif opt.phase == "eval":
        if opt.model_path != "":
            print("Loading model...\n")
            net.load_state_dict(torch.load(
                opt.net_path, map_location=lambda storage, loc: storage))
            g.load_state_dict(torch.load(
                opt.g_path, map_location=lambda storage, loc: storage))
            print("Loading model successful!")
            valid(testloader, g, net, criterion, device)
            print(f"\nAccuracy of the network on the test images: {accuracy:.2f}%.\n")
        else:
            print(
                "WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH")
    else:
        # print(opt)
        # print(trainset.filename.shape)
        # print(testset.filename.shape)
        for i, (input, target) in enumerate(trainloader):
            input, target = input.to(device), target.to(device)
            print(input.shape, target.shape)
