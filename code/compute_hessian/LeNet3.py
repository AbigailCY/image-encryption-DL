import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, bias=False)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)

        self.fc1 = nn.Linear(16*4*4, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

        self.weight_init()

    def forward(self, x):
        self.batch_size = x.shape[0]
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        self.x_1 = x.view(-1, 16*4*4)
        self.h_2 = self.fc1(self.x_1)
        self.x_2 = F.relu(self.h_2)
        self.h_3 = self.fc2(self.x_2)
        self.x_3 = F.relu(self.h_3)
        self.h_4 = self.fc3(self.x_3)
        with torch.no_grad():
            self.s = F.softmax(self.h_4, dim=1)
        return self.h_4

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weigth.data.fill_(1)
                m.bias.data.zero_()

    def compute_hessian(self):
        h4_hessian = self.s*(1-self.s)/self.batch_size

        W_3 = self.fc3.weight                                       # (10,84)
        B_3 = self.grad_relu(self.h_3)                              # (1,84)
        D_3 = self.hessian_relu(self.h_3)                           # (1,84)
        h3_hessian = self.recursive_pre_hessian(B_3, W_3, h4_hessian, D_3)  # (1,84)
        w3_hessian = self.kronecker(torch.pow(self.x_3, 2).t(), h4_hessian).t() # (10,84)

        W_2 = self.fc2.weight                                       # (84,120)
        B_2 = self.grad_relu(self.h_2)                              # (120,1)
        D_2 = self.hessian_relu(self.h_2)                           # (120,1)
        h2_hessian = self.recursive_pre_hessian(B_2, W_2, h3_hessian, D_2)  # (120,1)
        w2_hessian = self.kronecker(torch.pow(self.x_2, 2).t(), h3_hessian).t() # (84,120)

        a = 1

        # W_1 = self.fc1.weight
        # B_1 = self.grad_relu(self.h_1).view(-1, 1)
        # D_1 = self.hessian_relu(self.h_1).view(-1, 1)
        # h1_hessian = self.recursive_pre_hessian(B_1, W_1, h2_hessian, D_1)
        # w1_hessian = self.kronecker(torch.pow(self.x_1, 2).t(), h2_hessian)

        a = 1

    def kronecker(self, A, B):
        AB = torch.einsum("ab,cd->acbd", A, B)
        AB = AB.view(A.size(0)*B.size(0), A.size(1)*B.size(1))
        return AB

    def recursive_pre_hessian(self, B, W, H2, D):
        H1 = torch.pow(B, 2) * H2.matmul(torch.pow(W, 2)) + D
        return H1

    def grad_relu(self, x):
        return (x>0).float()

    def hessian_relu(self, x):
        return x*0