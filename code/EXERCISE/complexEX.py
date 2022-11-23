# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ComplexNN import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def encode(a, b, theta):
    x = torch.mul(a, torch.cos(theta)) - torch.mul(b, torch.sin(theta))
    y = torch.mul(a, torch.sin(theta)) + torch.mul(b, torch.cos(theta))
    # print(theta)
    return x, y

def decode(a, b, theta):
    return torch.mul(a, torch.cos(-theta)) - torch.mul(b, torch.sin(-theta))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding = 1)   
        self.conv2 = nn.Conv2d(64, 64, 3, padding = 1)  
        self.conv3 = ComplexConv2d(64, 128, 3, padding = 1)
        self.conv4 = ComplexConv2d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding = 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        # self.avgpool = nn.AvgPool2d(2, 2)
        self.globalavgpool = nn.AvgPool2d(8, 8)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = ComplexBatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout50 = nn.Dropout(0.5)
        self.dropout10 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 10)

        self.CReLU = ComplexRelu()
        self.CMax = ComplexMaxPool2d(kernel_size=2, stride=2)
        

    def forward(self, x, b, theta):             #100 3 32 32
        x = self.bn1(F.relu(self.conv1(x)))     #100 64 32 32
        x = self.bn1(F.relu(self.conv2(x)))     #100 64 32 32
        x = self.maxpool(x)                     #100 64 16 16
        x = self.dropout10(x)

        x1, x2 = encode(x, b, theta)
        x1, x2 = self.conv3(x1, x2)
        # print((x1*x1 + x2*x2).sqrt())
        x1, x2 = self.CReLU(x1, x2)
        x1, x2 = self.conv4(x1, x2)
        x1, x2 = self.CReLU(x1, x2)
        x1, x2 = self.bn2(x1, x2)
        x1, x2 = self.CMax(x1, x2)
        x = decode(x1, x2, theta)



        x = self.dropout10(x)
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.globalavgpool(x)
        x = self.dropout50(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# net = Net()
net = torch.load('cifar10.pkl')


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

b = torch.rand(100, 64, 16, 16)
theta = torch.Tensor(1).uniform_(0,2*np.pi)

for epoch in range(5):

    running_loss = 0.
    batch_size = 100
    
    for i, data in enumerate(
            torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2), 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs, b, theta)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))

print('Finished Training')

torch.save(net, 'cifar10.pkl')
# net = torch.load('cifar10.pkl')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images, b, theta)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images, b, theta)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))