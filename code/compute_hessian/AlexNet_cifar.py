import torch
from torch import nn
from torch.nn import functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight_conv1 = nn.Parameter(torch.Tensor(64, 3, 3, 3))
        self.bias_conv1 = None
        self.weight_conv2 = nn.Parameter(torch.Tensor(192, 64, 3, 3))
        self.bias_conv2 = None
        self.weight_conv3 = nn.Parameter(torch.Tensor(384, 192, 3, 3))
        self.bias_conv3 = None
        self.weight_conv4 = nn.Parameter(torch.Tensor(256, 384, 3, 3))
        self.bias_conv4 = None
        self.weight_conv5 = nn.Parameter(torch.Tensor(256, 256, 3, 3))
        self.bias_conv5 = None

        self.weight_fc1 = nn.Parameter(torch.Tensor(4096, 256))
        self.bias_fc1 = None
        self.weight_fc2 = nn.Parameter(torch.Tensor(4096, 4096))
        self.bias_fc2 = None
        self.weight_fc3 = nn.Parameter(torch.Tensor(10, 4096))
        self.bias_fc3 = None

        self.weight_init()

    def forward(self, x):
        self.conv1 = F.conv2d(x, self.weight_conv1, self.bias_conv1, stride=2, padding=1)                        # (63,6,24,24)
        self.conv1_pool, self.inds_pool1 = F.max_pool2d_with_indices(self.conv1, (2, 2))                         # (63,6,12,12)
        self.conv2 = F.conv2d(F.relu(self.conv1_pool), self.weight_conv2, self.bias_conv2, stride=2, padding=1)  # (63,16,8,8)
        self.conv2_pool, self.inds_pool2 = F.max_pool2d_with_indices(self.conv2, (2, 2))                         # (63,16,4,4)
        self.conv3 = F.conv2d(F.relu(self.conv2_pool), self.weight_conv3, self.bias_conv3, stride=1, padding=1)  # (63,16,8,8)       
        self.conv4 = F.conv2d(F.relu(self.conv3), self.weight_conv4, self.bias_conv4, stride=1, padding=1)  # (63,16,8,8) 
        self.conv5 = F.conv2d(F.relu(self.conv4), self.weight_conv5, self.bias_conv5, stride=1, padding=1)  # (63,16,8,8)
        self.conv5_pool, self.inds_pool5 = F.max_pool2d_with_indices(self.conv5, (2, 2))                         # (63,16,4,4)
        # print(self.conv5_pool.shape)


        self.x_1 = F.relu(self.conv5_pool).view(-1, 256*1*1)
        self.h_2 = F.linear(self.x_1, self.weight_fc1, self.bias_fc1)
        self.x_2 = F.relu(self.h_2)
        self.h_3 = F.linear(self.x_2, self.weight_fc2, self.bias_fc2)
        self.x_3 = F.relu(self.h_3)
        self.h_4 = F.linear(self.x_3, self.weight_fc3, self.bias_fc3)

        # print(self.conv1.shape, self.conv1_pool.shape, self.conv2.shape, self.conv2_pool.shape, self.conv3.shape, self.conv4.shape, self.conv5.shape, self.conv5_pool.shape, self.x_1.shape, self.x_2.shape, self.x_3.shape, self.h_4.shape)

        return self.h_4

    def weight_init(self):
        for w in self.parameters():
            nn.init.xavier_uniform_(w, gain=1)

    def compute_hessian(self, x):
        bs = x.shape[0]
        x_unfold = F.unfold(x, self.weight_conv1.shape[2:4], stride=(2, 2), padding=(1, 1))
        x_unfold = x_unfold.transpose(1, 2)     #[64, 256, 27]

        conv1_unfold = F.unfold(self.conv1_pool,  self.weight_conv2.shape[2:4], stride=(2, 2), padding=(1, 1))  # (63,150,64)
        conv1_unfold = conv1_unfold.transpose(1, 2)
        conv1_relu = F.relu(conv1_unfold)       #[64, 16, 576]

        conv2_unfold = F.unfold(self.conv2_pool,  self.weight_conv3.shape[2:4], stride=(1, 1), padding=(1, 1))  # (63,150,64)
        conv2_unfold = conv2_unfold.transpose(1, 2)
        conv2_relu = F.relu(conv2_unfold)       #[64, 4, 1728]

        conv3_unfold = F.unfold(self.conv3,  self.weight_conv4.shape[2:4], stride=(1, 1), padding=(1, 1))  # (63,150,64)
        conv3_unfold = conv3_unfold.transpose(1, 2)
        conv3_relu = F.relu(conv3_unfold)       #[64, 4, 3456]

        conv4_unfold = F.unfold(self.conv4,  self.weight_conv5.shape[2:4], stride=(1, 1), padding=(1, 1))  # (63,150,64)
        conv4_unfold = conv4_unfold.transpose(1, 2)
        conv4_relu = F.relu(conv4_unfold)       #[64, 4, 2304]

        ## ----------------------------------
        s = F.softmax(self.h_4, dim=1)
        h_hessian = s * (1 - s) / bs

        W = self.weight_fc3  # 
        h = self.h_3
        x = self.x_3
        fc3_hessian = h_hessian.t().matmul(torch.pow(x, 2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')  # 

        W = self.weight_fc2  # 
        h = self.h_2
        x = self.x_2
        fc2_hessian = h_hessian.t().matmul(torch.pow(x, 2))  
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')  # 

        W = self.weight_fc1  # 
        h = self.conv5_pool.view(bs, -1) 
        x = self.x_1
        fc1_hessian = h_hessian.t().matmul(torch.pow(x, 2))  
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')  # 

        W = self.weight_conv5
        c_out, c_in, k, k = W.shape
        W = W.view(c_out, c_in*k*k)
        h_hessian = h_hessian.view(self.conv5_pool.shape)
        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool5, (2, 2)).view(bs, c_out, -1).transpose(1,2).reshape(-1, c_out)
        h = conv4_unfold.reshape(-1, c_in*k*k)
        x = conv4_relu.reshape(-1, c_in*k*k)
        conv5_hessian = h_hessian.t().matmul(torch.pow(x, 2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')

        output_size = self.conv4.shape[2:4]
        kernel_size = (k, k)
        h_hessian = h_hessian.view(bs, -1, c_in*k*k).transpose(1,2)  
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=(1, 1), padding=(1, 1)) 

        W = self.weight_conv4
        c_out, c_in, k, k = W.shape
        W = W.view(c_out, c_in*k*k)
        h_hessian = h_hessian.view(self.conv4.shape).reshape(-1, c_out)
        h = conv3_unfold.reshape(-1, c_in*k*k)
        x = conv3_relu.reshape(-1, c_in*k*k)
        conv4_hessian = h_hessian.t().matmul(torch.pow(x, 2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')

        output_size = self.conv3.shape[2:4]
        kernel_size = (k, k)
        h_hessian = h_hessian.view(bs, -1, c_in*k*k).transpose(1,2)  
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=(1, 1), padding=(1, 1)) 

        W = self.weight_conv3
        c_out, c_in, k, k = W.shape
        W = W.view(c_out, c_in*k*k)
        h_hessian = h_hessian.view(self.conv3.shape).reshape(-1, c_out)
        h = conv2_unfold.reshape(-1, c_in*k*k)
        x = conv2_relu.reshape(-1, c_in*k*k)
        conv3_hessian = h_hessian.t().matmul(torch.pow(x, 2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')

        output_size = self.conv2_pool.shape[2:4]
        kernel_size = (k, k)
        h_hessian = h_hessian.view(bs, -1, c_in*k*k).transpose(1,2)  
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=(1, 1), padding=(1, 1)) 

        W = self.weight_conv2
        c_out, c_in, k, k = W.shape
        W = W.view(c_out, c_in*k*k)  
        h_hessian = h_hessian.view(self.conv2_pool.shape)
        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool2, (2, 2)).view(bs, c_out, -1).transpose(1,2).reshape(-1, c_out)
        h = conv1_unfold.reshape(-1, c_in*k*k)  
        x = conv1_relu.reshape(-1, c_in*k*k)
        conv2_hessian = h_hessian.t().matmul(torch.pow(x, 2))  # 
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'relu')  # 

        output_size = self.conv1_pool.shape[2:4]
        kernel_size = (k, k)
        h_hessian = h_hessian.view(bs, -1, c_in*k*k).transpose(1,2)  
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=(2, 2), padding=(1, 1)) 

        W = self.weight_conv1
        c_out, c_in, k, k = W.shape
        W = W.view(c_out, c_in * k * k)  # 
        h_hessian = h_hessian.view(self.conv1_pool.shape)
        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool1, (2, 2)).view(bs, c_out, -1).transpose(1,2).reshape(-1, c_out)
        h = x_unfold.reshape(-1, c_in*k*k)  # 
        x = h
        conv1_hessian = h_hessian.t().matmul(torch.pow(x, 2))  # 
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian, 'identity')  # 

        # print(fc3_hessian.shape, fc2_hessian.shape, fc1_hessian.shape, conv5_hessian.shape, conv4_hessian.shape, conv3_hessian.shape, conv2_hessian.shape, conv1_hessian.shape)
        # [10,4096] [4096, 4096] [4096, 256] [256, 2304] [256, 3456] [384, 1728] [192, 576] [64, 27]
        return fc3_hessian, fc2_hessian, fc1_hessian, conv5_hessian, conv4_hessian, conv3_hessian, conv2_hessian, conv1_hessian

    def recursive_pre_hessian(self, W, h, H2, activation):
        if activation=='relu':
            B = (h>0).float()
            D = h*0
        elif activation=='identity':
            B = torch.ones_like(h).cuda()
            # B = torch.ones_like(h)
            D = h*0
        else:
            raise NotImplementedError

        H1 = B.pow(2) * H2.matmul(W.pow(2)) + D
        return H1


x = torch.rand(64, 3, 32, 32)
net = AlexNet()
# a = net(x)
# a = F.softmax(a, dim=1)
# print(a.shape)
# net.compute_hessian(x)
for name, param in net.named_parameters():
	print(name, '      ', param.size())