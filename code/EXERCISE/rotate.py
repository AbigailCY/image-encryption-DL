
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F 

import numpy as np

# NUM_TRAIN = 49000
# transform = T.Compose([
#                 T.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#             ])

# cifar10_train = dset.CIFAR10('./', train=True, download=True,
#                              transform=transform)
# loader_train = DataLoader(cifar10_train, batch_size=64, 
#                           sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

# cifar10_val = dset.CIFAR10('./', train=True, download=True,
#                            transform=transform)
# loader_val = DataLoader(cifar10_val, batch_size=64, 
#                         sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

# cifar10_test = dset.CIFAR10('./', train=False, download=True, 
#                             transform=transform)
# loader_test = DataLoader(cifar10_test, batch_size=64)


# USE_GPU = True

# dtype = torch.float32 # we will be using float throughout this tutorial

# if USE_GPU and torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# # Constant to control how frequently we print train loss
# print_every = 100

# # print('using device:', device)

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


#exp(i theta)*(a+bi) = 
def encode(a, b, theta):
    x = torch.mul(a, torch.cos(theta)) - torch.mul(b, torch.sin(theta))
    y = torch.mul(a, torch.sin(theta)) + torch.mul(b, torch.cos(theta))
    # print(theta)
    return x, y

def decode(a, b, theta):
    return torch.mul(a, torch.cos(-theta)) - torch.mul(b, torch.sin(-theta))

class ComplexConv2d(nn.Module):

    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 0, 
                        dilation=1, groups=1, bias=False):

        super(ComplexConv2d, self).__init__()

        self.conv_w = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


    def forward(self,input_r,input_i):
        return self.conv_w(input_r), self.conv_w(input_i)

class ComplexRelu(nn.Module):

    def __init__(self):
        super(ComplexRelu, self).__init__()
    
    def forward(self, x1, x2, c):
        c1 = (x1*x1 + x2*x2).sqrt()
        non_Linear = c1 / torch.clamp(c1, max = c)
        return torch.mul(x1, non_Linear), torch.mul(x2, non_Linear)

class ComplexAveragePool(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, 
                 count_include_pad=True, divisor_override=None):
        super(ComplexAveragePool, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x1, x2):
        return F.avg_pool2d(x1, kernel_size, stride, padding, ceil_mode, 
                count_include_pad, divisor_override), \
               F.avg_pool2d(x2, kernel_size, stride, padding, ceil_mode, 
                count_include_pad, divisor_override)


class ComplexDropOut(nn.Module):
    def __init__(self):
        super(ComplexDropOut, self).__init__()
        
    def forward(self, )

class ComplexBN(nn.Module):
    def __init__(self):
        super(ComplexBN, self).__init__()




class ComplexMaxPool(nn.Module):
    def __init__(self):
        super(ComplexMaxPool, self).__init__()

class Net(nn.Module) : 
    def __init__(self):
        super(Net, self).__init__()

        # encoder
        self.local1 = nn.Sequential(                              # 3*32*32    # 3*227*227
            nn.Conv2d(in_channels=3, out_channels=64,kernel_size=11,stride=4,padding=2, bias=False),#in 3*223*223   out 64*55*55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),   

            nn.Conv2d(in_channels=64,out_channels=192,kernel_size=5,padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2), 

            nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.rotate = None

        # process

        self.conv4=nn.Sequential(
            ComplexConv2d(in_channels=384,out_channels=256,kernel_size=3, padding=1, bias = False),
            nn.ReLU(inplace=True)
        )
        self.conv5=nn.Sequential(
            ComplexConv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1, bias = False),
            nn.BatchNorm2d(256)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
            nn.Dropout()
        )

        #decoder

        self.derotate = None

        self.local2 = n.Sequential(
            nn.Dropout(),
            nn.Liner(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Liner(4096,4096),
            nn.ReLU(inplace=True),
            nn.Liner(4096,10)
    )


    def forward(self, x, b, theta):

        x = self.local1(x)
        x1, x2 = encode(x, b, theta)



        x = decode(x1, x2, theta)


        return x