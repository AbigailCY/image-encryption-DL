import torch
from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight_conv1 = nn.Parameter(torch.zeros((6, 1, 5, 5)))
        self.bias_conv1 = None

        self.weight_conv2 = nn.Parameter(torch.zeros((16, 6, 5, 5)))
        self.bias_conv2 = None

        self.weight_fc1 = nn.Parameter(torch.zeros((120, 16 * 4 * 4)))
        self.bias_fc1 = None
        self.weight_fc2 = nn.Parameter(torch.zeros((84, 120)))
        self.bias_fc2 = None
        self.weight_fc3 = nn.Parameter(torch.zeros((10, 84)))
        self.bias_fc3 = None

        self.weight_init()

    def forward(self, x):
        self.conv1 = F.conv2d(x, self.weight_conv1, self.bias_conv1)                        # (63,6,24,24)
        self.conv1_pool, self.inds_pool1 = F.max_pool2d_with_indices(self.conv1, (2, 2))    # (63,6,12,12)
        self.conv2 = F.conv2d(F.relu(self.conv1_pool), self.weight_conv2, self.bias_conv2)  # (63,16,8,8)
        self.conv2_pool, self.inds_pool2 = F.max_pool2d_with_indices(self.conv2, (2, 2))    # (63,16,4,4)

        self.x_1 = F.relu(self.conv2_pool).view(-1, 16 * 4 * 4)
        self.h_2 = F.linear(self.x_1, self.weight_fc1, self.bias_fc1)
        self.x_2 = F.relu(self.h_2)
        self.h_3 = F.linear(self.x_2, self.weight_fc2, self.bias_fc2)
        self.x_3 = F.relu(self.h_3)
        self.h_4 = F.linear(self.x_3, self.weight_fc3, self.bias_fc3)

        return self.h_4

    def weight_init(self):
        for w in self.parameters():
            nn.init.xavier_uniform_(w, gain=1)

    def compute_hessian(self, x):
        bs = x.shape[0]
        x_unfold = F.unfold(x, self.weight_conv1.shape[2:4], stride=(1, 1), padding=(0, 0))
        x_unfold = x_unfold.transpose(1, 2)

        conv1_unfold = F.unfold(self.conv1_pool,  self.weight_conv2.shape[2:4], stride=(1, 1), padding=(0, 0))  # (63,150,64)
        conv1_unfold = conv1_unfold.transpose(1, 2)
        conv1_relu = F.relu(conv1_unfold)

        ## ----------------------------------
        s = F.softmax(self.h_4, dim=1)
        h_hessian = s * (1 - s) / bs

        W = self.weight_fc3  # (10,84)
        h = self.h_3
        x = self.x_3
        fc3_hessian = h_hessian.t().matmul(torch.pow(x, 2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')  # (63,84)

        W = self.weight_fc2  # (84,120)
        h = self.h_2
        x = self.x_2
        fc2_hessian = h_hessian.t().matmul(torch.pow(x, 2))  # (84,120)
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')  # (63,120)

        W = self.weight_fc1  # (120,256)
        h = self.conv2_pool.view(bs, -1) # (63,256)
        x = self.x_1
        fc1_hessian = h_hessian.t().matmul(torch.pow(x, 2))  # (120,256)
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')  # (63,256)

        W = self.weight_conv2
        c_out, c_in, k, k = W.shape
        W = W.view(c_out, c_in*k*k)  # (16,150)
        h_hessian = h_hessian.view(self.conv2_pool.shape)
        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool2, (2, 2)).view(bs, c_out, -1).transpose(1,2).reshape(-1, c_out)
        h = conv1_unfold.reshape(-1, c_in*k*k)   # (63*64,150)
        x = conv1_relu.reshape(-1, c_in*k*k)
        conv2_hessian = h_hessian.t().matmul(torch.pow(x, 2))  # (16,150)
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')  # (63*64,150)

        output_size = self.conv1_pool.shape[2:4]
        kernel_size = (k, k)
        h_hessian = h_hessian.view(bs, -1, c_in*k*k).transpose(1,2)  # (63,150,64)
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=(1, 1), padding=(0, 0)) # (63,6,12,12)

        W = self.weight_conv1
        c_out, c_in, k, k = W.shape
        W = W.view(c_out, c_in * k * k)  # (16,150)
        h_hessian = h_hessian.view(self.conv1_pool.shape)
        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool1, (2, 2)).view(bs, c_out, -1).transpose(1,2).reshape(-1, c_out)
        h = x_unfold.reshape(-1, c_in*k*k)  # (63*576,25)
        x = h
        conv1_hessian = h_hessian.t().matmul(torch.pow(x, 2))  # (6,25)
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'identity')  # (63*576,25)

        # [10,84] [84, 120] [120, 256] [16, 150] [6, 25]
        return fc3_hessian, fc2_hessian, fc1_hessian, conv2_hessian, conv1_hessian

    def recursive_pre_hessian(self, W, h, H2, activation):
        if activation=='relu':
            B = (h>0).float()
            D = h*0
        elif activation=='identity':
            B = torch.ones_like(h).cuda()
            D = h*0
        else:
            raise NotImplementedError

        H1 = B.pow(2) * H2.matmul(W.pow(2)) + D
        return H1


