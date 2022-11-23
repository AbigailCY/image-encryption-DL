import torch
from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight_conv1 = nn.Parameter(torch.Tensor(6, 1, 5, 5))
        self.bias_conv1 = None
        self.weight_conv2 = nn.Parameter(torch.Tensor(16, 6, 5, 5))
        self.bias_conv2 = None

        self.weight_fc1 = nn.Parameter(torch.Tensor(120, 16*5*5))
        self.bias_fc1 = None
        self.weight_fc2 = nn.Parameter(torch.Tensor(84, 120))
        self.bias_fc2 = None
        self.weight_fc3 = nn.Parameter(torch.Tensor(10, 84))
        self.bias_fc3 = None

        self.weight_init()

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.a_1 = F.conv2d(x, self.weight_conv1, self.bias_conv1, padding = 2)
        self.b_1 = F.max_pool2d(self.a_1, (2, 2))
        self.c_1 = F.relu(self.b_1)
        self.a_2 = F.conv2d(self.c_1, self.weight_conv2, self.bias_conv2)
        self.b_2 = F.max_pool2d(self.a_2, (2, 2))
        self.h_1 = self.b_2.view(-1, 16*5*5)
        self.x_1 = F.relu(self.h_1)
        self.h_2 = F.linear(self.x_1, self.weight_fc1, self.bias_fc1)
        self.x_2 = F.relu(self.h_2)
        self.h_3 = F.linear(self.x_2, self.weight_fc2, self.bias_fc2)
        self.x_3 = F.relu(self.h_3)
        self.h_4 = F.linear(self.x_3, self.weight_fc3, self.bias_fc3)
        with torch.no_grad():
            self.s = F.softmax(self.h_4, dim=1)
        return self.h_4

    def weight_init(self):
        for w in self.parameters():
            nn.init.xavier_uniform_(w, gain=1)

    def compute_hessian(self):
        h4_hessian = self.s*(1-self.s)/self.batch_size

        W_3 = self.weight_fc3                                       # (10,84)
        B_3 = self.grad_relu(self.h_3)                              # (64,84)
        D_3 = self.hessian_relu(self.h_3)                           # (64,84)
        h3_hessian = self.recursive_pre_hessian(B_3, W_3, h4_hessian, D_3)  # (64,84)
        w3_hessian = h4_hessian.t().matmul(torch.pow(self.x_3, 2))  # (10,84)

        W_2 = self.weight_fc2                                       # (84,120)
        B_2 = self.grad_relu(self.h_2)                              # (64,120)
        D_2 = self.hessian_relu(self.h_2)                           # (64,120)
        h2_hessian = self.recursive_pre_hessian(B_2, W_2, h3_hessian, D_2)  # (64,120)
        w2_hessian = h3_hessian.t().matmul(torch.pow(self.x_2, 2))  # (84,120)

        W_1 = self.weight_fc1                                       # (120,400)
        B_1 = self.grad_relu(self.h_1)                              # (64,400)
        D_1 = self.hessian_relu(self.h_1)                           # (64,400)
        h1_hessian = self.recursive_pre_hessian(B_1, W_1, h2_hessian, D_1)  # (64,400)
        w1_hessian = h2_hessian.t().matmul(torch.pow(self.x_1, 2))  # (120,400)

        a = 1

        # return w1_hessian.view(-1), w2_hessian.view(-1), w3_hessian.view(-1)
        return w1_hessian, w2_hessian, w3_hessian

    def recursive_pre_hessian(self, B, W, H2, D):
        H1 = torch.pow(B, 2) * H2.matmul(torch.pow(W, 2)) + D
        return H1

    def grad_relu(self, x):
        return (x>0).float()

    def hessian_relu(self, x):
        return x*0


# x = torch.rand(64,1,28,28)
# net = LeNet()
# x = net(x)
# print(x.shape)
# net.compute_hessian()
# for name, param in net.named_parameters():
# 	print(name, '      ', param.size())