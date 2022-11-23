import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ComplexNN import *


class preEncoder(nn.Module):
    def __init__(self):
        super(preEncoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, 5, 1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, I):
        return self.layers(I)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(G_DIM*G_SIZE*G_SIZE, 2 * D_CHANNEL),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2 * D_CHANNEL, D_CHANNEL),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(D_CHANNEL, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.g = Encoder()
        self.conv2 = ComplexConv2d(6, 16, 5, 1)
        self.Crelu = ComplexRelu()
        self.Cmax = ComplexMaxPool2d(2, 2)

        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)


    def forward(self, a, b, theta):
  
        x1, x2 = encode(a, b, theta)
        x1, x2 = self.conv2(x1, x2)
        x1, x2 = self.Crelu(x1, x2)
        x1, x2 = self.Cmax(x1, x2)

        x1 = x1.view(x1.size()[0], -1)
        x2 = x2.view(x2.size()[0], -1)

        x1, x2 = self.fc1(x1), self.fc1(x2)
        x1, x2 = self.fc2(x1), self.fc2(x2)
        x1, x2 = self.fc3(x1), self.fc3(x2)

        x = decode(x1, x2, theta)
        return x

EPOCHS = 20
BATCH_SIZE = 100
LR = 0.00005           
D_CHANNEL = 256          # number of channels inner discriminator
CRITIC = 5              # number of training steps for discriminator per iter
CLIP_VALUE = 0.01       # lower and upper clip value for disc. weights
K_SYNTHETIC = 5
G_DIM = 6
G_SIZE = 14

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset = torchvision.datasets.CIFAR10(root='./', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./', train=False,
                                       download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# g = preEncoder()
# discriminator = Discriminator()
# net = Net()
g = torch.load('Lenet_CNN_g.pkl')
discriminator = torch.load('Lenet_CNN_D.pkl')
net = torch.load('Lenet_CNN.pkl')

criterion = nn.CrossEntropyLoss()
optimizer_g = optim.RMSprop(g.parameters(), lr=LR)
optimizer_D = optim.RMSprop(discriminator.parameters(), lr=LR)
optimizer = optim.Adam(net.parameters(), lr=0.002)

net = net.to(device)
g = g.to(device)
discriminator = discriminator.to(device)

# b = torch.rand(BATCH_SIZE, G_DIM, G_SIZE, G_SIZE).to(device)
theta = torch.Tensor(1).uniform_(0,2*np.pi).to(device) 
for epoch in range(EPOCHS):
    
    for i, data in enumerate(trainloader, 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        a = g(inputs).detach()
        b = g(torch.rand_like(inputs)).detach()
        bs = a.shape[0]

        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = True


        for j in range(CRITIC):
            discriminator.zero_grad()
            delta_theta = torch.Tensor(K_SYNTHETIC).uniform_(0,np.pi).to(device)
            fake_imgs = torch.stack([ED(a, b, dt) for dt in delta_theta])
            fake_imgs = fake_imgs.view(K_SYNTHETIC, bs, -1)

            loss_D = -torch.mean(discriminator(a.view(bs, -1))) + torch.mean(discriminator(fake_imgs))
            loss_D.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = False

        optimizer_g.zero_grad()
        a = g(inputs)
        delta_theta = torch.Tensor(1).uniform_(0,np.pi).to(device)
        fake_img = ED(a, b, delta_theta)
        fake_imgs = fake_img.view(bs, -1)
        loss_g = torch.mean(discriminator(a.view(bs, -1))) - torch.mean(discriminator(fake_imgs))
        loss_g.backward()
        optimizer_g.step()

        optimizer.zero_grad()
        optimizer_g.zero_grad()
        a = g(inputs)
        outputs = net(a, b, theta) #[100, 10] [100,6,14,14]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_g.step()
        optimizer.step()
        
        print('[%d, %5d] loss: %.4f, %.4f' %(epoch + 1, (i+1)*BATCH_SIZE, loss.item(), loss_g))

print('Finished Training')

torch.save(g, './check/Lenet_CNN_g.pkl')
torch.save(discriminator, './check/Lenet_CNN_D.pkl')
torch.save(net, './check/Lenet_CNN.pkl')
# net = torch.load('./check/Lenet_CNN.pkl')
# g = torch.load('./check/Lenet_CNN_g.pkl')

correct = 0
total = 0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

theta = torch.Tensor(1).uniform_(0,2*np.pi).to(device) 
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        b = torch.rand_like(images).to(device)
        b = g(b)
        a = g(images)
        outputs = net(a, b, theta)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        c = (predicted == labels).squeeze()
        for i in range(labels.shape[0]):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))