import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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

        # self.model = nn.Sequential(
        #     nn.Linear(G_DIM*G_SIZE*G_SIZE, 2*D_CHANNEL),
        #     # nn.ReLU(inplace=True),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(2 * D_CHANNEL, D_CHANNEL),
        #     # nn.ReLU(inplace=True),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(D_CHANNEL, 1),
        # )
        self.linear1 = nn.Linear(G_DIM*G_SIZE*G_SIZE, 2*D_CHANNEL)
        self.linear2 = nn.Linear(2 * D_CHANNEL, D_CHANNEL)
        self.linear3 = nn.Linear(D_CHANNEL, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # x = self.model(x)
        x = self.linear1(x)

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

G_DIM = 6
G_SIZE = 14
D_CHANNEL = 256
g = preEncoder()
discriminator = Discriminator()
net = Net()

inputs = torch.rand(64,3,32,32)

b = g(torch.rand_like(inputs))
a = g(inputs)
print(a.shape)

# a = torch.rand(64,6,14,14)
# b = discriminator(a)
# print(b.shape)

# theta = torch.Tensor(1).uniform_(0,2*np.pi)
# a = net(a,b,theta)
# print(a.shape)

# x = Variable(torch.Tensor([5,2]), requires_grad=True)
# y_0 = x[0]**2 + x[1]**2
# y_1 = x[0]**4 + x[1]**3

# grad_x_0 = torch.autograd.grad(y_0, x, create_graph=True)
# print(grad_x_0)

# grad_x_1 = torch.autograd.grad(y_1, x, create_graph=True)
# print(grad_x_1)

# grad_grad_x_0 = torch.autograd.grad([grad_x_0[0][0],grad_x_0[0][1]], x, create_graph=True)
# print(grad_grad_x_0)

# grad_grad_x_1 = torch.autograd.grad([grad_x_1[0][0],grad_x_1[0][1]], x, create_graph=True)
# print(grad_grad_x_1)

# class AlexNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.weight_conv1 = nn.Parameter(torch.Tensor(64, 3, 3, 3))
#         self.bias_conv1 = None
#         self.weight_conv2 = nn.Parameter(torch.Tensor(192, 64, 3, 3))
#         self.bias_conv2 = None
#         self.weight_conv3 = nn.Parameter(torch.Tensor(384, 192, 3, 3))
#         self.bias_conv3 = None
#         self.weight_conv4 = nn.Parameter(torch.Tensor(256, 384, 3, 3))
#         self.bias_conv4 = None
#         self.weight_conv5 = nn.Parameter(torch.Tensor(256, 256, 3, 3))
#         self.bias_conv5 = None

#         self.weight_fc1 = nn.Parameter(torch.Tensor(4096, 256))
#         self.bias_fc1 = None
#         self.weight_fc2 = nn.Parameter(torch.Tensor(4096, 4096))
#         self.bias_fc2 = None
#         self.weight_fc3 = nn.Parameter(torch.Tensor(10, 4096))
#         self.bias_fc3 = None

#         self.weight_init()

#     def forward(self, x):
#         self.batch_size = x.shape[0]
#         self.a_1 = F.conv2d(x, self.weight_conv1, self.bias_conv1, stride=2, padding=1)
#         print(self.a_1.shape)
#         self.b_1 = F.max_pool2d(self.a_1, (2, 2))
#         print(self.b_1.shape)
#         self.c_1 = F.relu(self.b_1)
#         self.a_2 = F.conv2d(self.c_1, self.weight_conv2, self.bias_conv2, stride=2, padding=1)
#         print(self.a_2.shape)
#         self.b_2 = F.max_pool2d(self.a_2, (2, 2))
#         print(self.b_2.shape)
#         self.c_2 = F.relu(self.b_2)
#         self.a_3 = F.conv2d(self.c_2, self.weight_conv3, self.bias_conv3, stride=1, padding=1)
#         print(self.a_3.shape)
#         # self.b_2 = F.max_pool2d(self.a_2, (2, 2))
#         self.c_3 = F.relu(self.a_3)
#         self.a_4 = F.conv2d(self.c_3, self.weight_conv4, self.bias_conv4, stride=1, padding=1)
#         print(self.a_4.shape)
#         # self.b_2 = F.max_pool2d(self.a_2, (2, 2))
#         self.c_4 = F.relu(self.a_4)
#         self.a_5 = F.conv2d(self.c_4, self.weight_conv5, self.bias_conv5, stride=1, padding=1)
#         print(self.a_5.shape)
#         self.b_5 = F.max_pool2d(self.a_5, (2, 2))
#         self.c_5 = F.relu(self.b_5)
#         print(self.c_5.shape)
#         self.h_1 = self.c_5.view(64, -1)
#         print(self.h_1.shape)
#         # droupout
#         self.d_1 = F.dropout(self.h_1)
#         # self.x_1 = F.relu(self.d_1)
#         self.h_2 = F.linear(self.d_1, self.weight_fc1, self.bias_fc1)
#         self.x_2 = F.relu(self.h_2)
#         self.d_2 = F.dropout(self.x_2)
#         self.h_3 = F.linear(self.d_2, self.weight_fc2, self.bias_fc2)
#         self.x_3 = F.relu(self.h_3)
#         self.h_4 = F.linear(self.x_3, self.weight_fc3, self.bias_fc3)
#         with torch.no_grad():
#             self.s = F.softmax(self.h_4, dim=1)
#         return self.h_4
#     def weight_init(self):
#         for w in self.parameters():
#             nn.init.xavier_uniform_(w, gain=1)
# net = AlexNet()
# x = torch.rand(64,3,32,32)
# net(x)
