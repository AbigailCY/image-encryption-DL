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

from utils import *

parser = argparse.ArgumentParser(description='CelebA Alexnet')
parser.add_argument('--dataroot', type=str,
                    default='/Extra/dataset/face/CelebA/Img', help="celeba dataset path.")
parser.add_argument('--datasets', type=str, default="celeba", help="dataset")
parser.add_argument('--batch_size', type=int, default=128,
                    help="Every train dataset size.")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="starting lr, every 10 epoch decay 10.")
parser.add_argument('--epochs', type=int, default=200, help="Train loop")
parser.add_argument('--phase', type=str, default='eval',
                    help="train or eval? Default:`eval`")
parser.add_argument('--model_path', type=str,
                    default="./check/celeba_alexnet/celeba_epoch_18.pth", help="load model path.")
# ./check/celeba_alexnet/celeba_epoch_3.pth
opt = parser.parse_args()
print(opt)

try:
    os.makedirs("./check/celeba_alexnet")
except OSError:
    pass

manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True


class AlexNet(nn.Module):

    def __init__(self, identities=10177):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, identities),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


transform = transforms.Compose([
    #    transforms.Resize(image_size),
    #    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CelebA(root=opt.dataroot,  split='train',
                                       target_type='identity', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                          shuffle=True, num_workers=2)
# 19867
valset = torchvision.datasets.CelebA(root=opt.dataroot,  split='valid',
                                     target_type='identity', transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size,
                                        shuffle=True, num_workers=2)
# 19962
testset = torchvision.datasets.CelebA(root=opt.dataroot,  split='test',
                                      target_type='identity', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = AlexNet().to(device)
# model.load_state_dict(torch.load(
#     opt.model_path, map_location=lambda storage, loc: storage))
# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)


def train(train_dataloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_dataloader):

        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)-1
        # compute output
        output = model(inputs)
        loss = criterion(output, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        # top5.update(prec5, inputs.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 5 == 0:
            print(f"Epoch [{epoch + 1}] [{i}/{len(train_dataloader)}]\t"
                  f"Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  f"Loss {loss.item():.4f}\t"
                  # f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
                  f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t", end='\r')
    if epoch % 10 == 9:
        torch.save(model.state_dict(),
               f"./check/celeba_alexnet/{opt.datasets}_epoch_{epoch + 1}.pth")


def test(model, dataloader):
    # switch to evaluate mode
    model.eval()
    # init value
    total = 0.
    correct = 0.
    top1 = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)-1

            outputs = model(inputs)
            prec1 = accuracy(outputs, targets, topk=(1,))
            top1.update(prec1[0], inputs.shape[0])

            # _, predicted = torch.max(outputs.data, 1)
            # total += targets.size(0)
            # correct += (predicted == targets).sum().item()

    # accuracy = 100 * correct / total
    return top1.avg


def run():
    best_prec1 = 0.
    for epoch in range(19, opt.epochs):
        # train for one epoch
        print(f"\nBegin Training Epoch {epoch + 1}")
        train(trainloader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        print(f"Begin Validation @ Epoch {epoch + 1}")
        prec1 = test(model, valloader)

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
            model.load_state_dict(torch.load(
                opt.model_path, map_location=lambda storage, loc: storage))
            print("Loading model successful!")
            accuracy = test(model, testloader)
            print(
                f"\nAccuracy of the network on the test images: {accuracy:.2f}%.\n")
        else:
            print(
                "WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH")
    else:
        print(opt)
        print(trainset.filename.shape)
        print(valset.filename.shape)
        print(testset.filename.shape)
