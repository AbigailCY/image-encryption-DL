import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ComplexNN import *

class AlexNet(nn.Module):

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # x = self.features(x)
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = self.conv3(x)
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
# net = AlexNet()
# x = torch.rand(64,3,32,32)
# net(x)
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
        # print(x.shape)
        x = x.view(x.shape[0], -1)  #[128, 256]
        # print(x.shape)
        x = self.linear(x)
        # print(x.shape)
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

g = preEncoder()
discriminator = Discriminator()
net = Net()
# model = AlexNet()
x = torch.rand(128, 3, 218, 178)
# b = g(torch.rand_like(x))
x = g(x)
print(x.shape)
# theta = torch.Tensor(1).uniform_(0,np.pi)
# x = net(x,b,theta)
x = discriminator(x)
print(x.shape)
# a = g(x)
# b = g(torch.rand_like(x))
# delta_theta = torch.Tensor(5).uniform_(0,np.pi)
# output = net(a, b, delta_theta[0]) 
# print(ED(a, b, delta_theta[0]).shape)
# print(discriminator(ED(a, b, delta_theta[0])))
# fake_imgs = torch.stack([discriminator(ED(a, b, dt)) for dt in delta_theta])
# loss_D = -torch.mean(discriminator(a)) + torch.mean(fake_imgs)
# print(x.shape)
# x = model(x)
# print(x.shape)
# x = d(x)
# print(x.shape)
# 
# target = torch.ones([256, 40], dtype=torch.int64).float()  # 64 classes, batch size = 10
# output = torch.full([256, 40], 0.999)  # A prediction (logit)
# pos_weight = torch.ones([40])  # All weights are equal to 1
# criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# loss = criterion(output, target)  # -log(sigmoid(0.999))
# print(loss)
# def encode(a, b, theta):
#     x = torch.mul(a, torch.cos(theta)) - torch.mul(b, torch.sin(theta))
#     y = torch.mul(a, torch.sin(theta)) + torch.mul(b, torch.cos(theta))
#     # print(theta)
#     return x, y

# def decode(a, b, theta):
#     return torch.mul(a, torch.cos(-theta)) - torch.mul(b, torch.sin(-theta))



# print(x.shape[0])
# b = g(torch.rand_like(x))
# x = g(x)

# print(x.shape)

# theta = torch.Tensor(1).uniform_(0,2*np.pi)
# output = net(x, b, theta)
# print(output.shape)
# def encode1(a, b, theta):
#     x = a * torch.cos(theta) - b * torch.sin(theta)
#     y = a * torch.sin(theta) + b * torch.cos(theta)
#     # print(theta)
#     return x, y


# class preEncoder(nn.Module):
#     def __init__(self):
#         super(preEncoder, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, I):
#         return self.features(I)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.main = nn.Sequential(
#             nn.Conv2d(384,192,3,1),
#             nn.LeakyReLU(),
#             nn.Conv2d(192,64,2,1),
#             nn.LeakyReLU(),
#         )
#         self.linear = nn.Linear(64, 1)
#         self.relu = nn.LeakyReLU()

#     def forward(self, x):
#         x = self.main(x)
#         x = x.view(x.shape[0], -1)
#         x = self.linear(x)
#         x = self.relu(x)

#         return x

# # g = preEncoder()
# d = Discriminator()
# x = torch.rand(10,384,4,4)
# # x = g(x)
# # print(x.shape)
# x = d(x)
# print(x.shape)

# print(3e-4)



# a = torch.tensor([2])
# b = torch.tensor([3])

# b = torch.rand(1, 6, 1, 2)
# # print(b)

# pool = nn.MaxPool2d(2, stride=2, return_indices=True)
# unpool = nn.MaxUnpool2d(2, stride=2)
# Input = torch.tensor([[[[ 1.,  2,  3,  4],
#                         [ 5,  6,  7,  8],
#                         [ 9, 10, 11, 12],
#                         [13, 14, 15, 16]],[[ 1.,  2,  3,  4],
#                         [ 5,  6,  7,  8],
#                         [ 9, 10, 11, 12],
#                         [13, 14, 15, 16]]],[[[ 0.,  0,  0,  0],
#                         [ 0,  1,  0,  1],
#                         [ 0,0,0,0],
#                         [3,4,0,1]],[[ 0.,  0,  0,  0],
#                         [ 0,  1,  0,  1],
#                         [ 0,0,0,0],
#                         [0,1,0,1]]]])
# output, indices = pool(Input)

# I = torch.tensor([[[[ 0.,  0,  0,  0],
#                         [ 0,  1,  0,  1],
#                         [ 0,0,0,0],
#                         [0,1,0,1]]]])
# def unravel_index(index, shape):
#     out = []
#     for dim in reversed(shape):
#         out.append(index % dim)
#         index = index // dim
#     return tuple(reversed(out))
# print(Input)
# print(torch.mean(Input,(0)))

# A  = torch.tensor([[ 0.,  1,  0,  1],[ 0,0,0,0],[0,1,0,1]])
# A[0,:] = 3
# print(A)
# print(torch.mean(A))
                        
                        
    
# print(unravel_index(1, Input.shape))
# print(Input[unravel_index(1, Input.shape)])
# # Input[15] = 100
# y = Input.view(16)
# indices[0,0,0,1] = y[indices[0,0,0,1]]


# theta = torch.Tensor(10).uniform_(0,2*np.pi)
# b = torch.rand(100, 6, 14, 14)
# print(b)

# G_DIM = 6
# G_SIZE = 14
# D_CHANNEL=256
# x = torch.rand(10,100,6,14,14)

# model = nn.Sequential(
#         nn.Linear(G_DIM*G_SIZE*G_SIZE, 256),
#         nn.LeakyReLU(0.2),
#         nn.Linear(256, 128),
#         nn.LeakyReLU(0.2),
#         nn.Linear(128, 1),
#     )


# x = x.view(x.shape[0],x.shape[1],G_DIM*G_SIZE*G_SIZE)
# x = torch.mean(model(x))
# print(x.shape)
# x = torch.rand(50)
# print(x.shape)
# for i in range(x.shape[0]):
#     print(i)

# def ED(a, b, theta):
#     x, y = encode(a, b, theta)
#     return decode(x, y, theta)
# a = torch.rand(100,3,14,14)
# b = torch.rand(100,3,14,14)
# theta = torch.Tensor(10).uniform_(0,2*np.pi)
# x = torch.stack([ED(a, b, dt) for dt in theta])

# print(x.shape)



# a = torch.tensor([[1.,2.],[3.,4.]])
# non_Linear = torch.clamp(a, max = 2)

# print(non_Linear)
# non_Linear = a / torch.clamp(a, max = 2)
# print(a)
# print(non_Linear)

# x1 = torch.rand([100, 3, 32, 32])
# x_norm_max, inds = F.max_pool2d(x1, 2, 2, return_indices = True)
# print(inds[1,1,:,:])
# print(a_max.shape, "\n")
# print(inds.shape, "\n")
# n,c,h,w = inds.shape
# b = a.view(100,3,32*32)
# inds_b = inds.view(n,c,h*w)
# b_max = torch.gather(b, 2, inds_b)
# print(b_max.shape, "\n")
# b_max = b_max.view(n,c,h,w)
# print(b_max.shape, "\n")

# x1 = torch.rand([100, 3, 32, 32])
# x2 = torch.rand([100, 3, 32, 32])
# x_norm = x1*x1 + x2*x2
# x_norm_max, inds = F.max_pool2d(x_norm, 2, 2, return_indices = True)

# x1 = torch.rand([100, 3, 32, 32])
# x2 = torch.rand([100, 3, 32, 32])
# x_norm = x1*x1 + x2*x2
# x_norm_max, inds = F.max_pool2d(x_norm, 2, 2, 0, 1, False, True)
# n,c,h1,w1 = x_norm.shape
# n,c,h2,w2 = x_norm_max.shape
# X1 = x1.view(n,c,h1*w1)
# X2 = x2.view(n,c,h1*w1)
# inds_X = inds.view(n,c,h2*w2)
# X1_max = X1.gather(2, inds_X).view(n,c,h2,w2)
# X2_max = X2.gather(2, inds_X).view(n,c,h2,w2)


# x1 = torch.rand([100, 3, 32, 32])
# x2 = torch.rand(x1.shape)
# print(x2.shape)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs of training")
# parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
# parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
# # parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
# # parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
# parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
# parser.add_argument("--channels", type=int, default=3, help="number of image channels")
# parser.add_argument("--n_critic", type=int, default=10, help="number of training steps for discriminator per iter")
# parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
# # parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
# opt = parser.parse_args()
# opt.channels = 64
# img_shape = (opt.channels, opt.img_size, opt.img_size)
# print(img_shape)


# a = torch.rand(3,5,64,64)
# b = nn.Conv2d(5,6,5,1)
# # b = torch.rand_like(a)

# print(b)






