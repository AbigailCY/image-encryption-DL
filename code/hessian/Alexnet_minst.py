import torch
from torch import nn
from torch.nn import functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight_conv1 = nn.Parameter(torch.Tensor(32, 1, 3, 3))
        self.bias_conv1 = None
        self.weight_conv2 = nn.Parameter(torch.Tensor(64, 32, 3, 3))
        self.bias_conv2 = None
        self.weight_conv3 = nn.Parameter(torch.Tensor(128, 64, 3, 3))
        self.bias_conv3 = None
        self.weight_conv4 = nn.Parameter(torch.Tensor(256, 128, 3, 3))
        self.bias_conv4 = None
        self.weight_conv5 = nn.Parameter(torch.Tensor(256, 256, 3, 3))
        self.bias_conv5 = None

        self.weight_fc1 = nn.Parameter(torch.Tensor(1024, 256*3*3))
        self.bias_fc1 = None
        self.weight_fc2 = nn.Parameter(torch.Tensor(512, 1024))
        self.bias_fc2 = None
        self.weight_fc3 = nn.Parameter(torch.Tensor(10, 512))
        self.bias_fc3 = None

        self.weight_init()

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.a_1 = F.conv2d(x, self.weight_conv1, self.bias_conv1, padding=1)
        self.b_1 = F.max_pool2d(self.a_1, (2, 2))
        self.c_1 = F.relu(self.b_1)

        self.a_2 = F.conv2d(self.c_1, self.weight_conv2, self.bias_conv2, stride=1, padding=1)
        self.b_2 = F.max_pool2d(self.a_2, (2, 2))
        self.c_2 = F.relu(self.b_2)

        self.a_3 = F.conv2d(self.c_2, self.weight_conv3, self.bias_conv3, stride=1, padding=1)
        self.c_3 = F.relu(self.a_3)

        self.a_4 = F.conv2d(self.c_3, self.weight_conv4, self.bias_conv4, stride=1, padding=1)
        self.c_4 = F.relu(self.a_4)

        self.a_5 = F.conv2d(self.c_4, self.weight_conv5, self.bias_conv5, stride=1, padding=1)
        self.b_5 = F.max_pool2d(self.a_5, (2, 2))
        self.c_5 = F.relu(self.b_5)

        self.h_1 = self.c_5.view(-1, 256*3*3)

        # 去掉两个droupout层， 第一个droupout用relu代替
        # self.d_1 = F.dropout(self.h_1)
        self.x_1 = F.relu(self.h_1)
        self.h_2 = F.linear(self.x_1, self.weight_fc1, self.bias_fc1)
        self.x_2 = F.relu(self.h_2)
        # self.d_2 = F.dropout(self.x_2)
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

        W_3 = self.weight_fc3                                       # (10,512)
        B_3 = self.grad_relu(self.h_3)                              # (64,512)
        D_3 = self.hessian_relu(self.h_3)                           # (64,512)
        h3_hessian = self.recursive_pre_hessian(B_3, W_3, h4_hessian, D_3)  # (10,512)
        w3_hessian = h4_hessian.t().matmul(torch.pow(self.x_3, 2))
        # print(B_3.shape, h3_hessian.shape, w3_hessian.shape)

        W_2 = self.weight_fc2                                       # (512, 1024)
        B_2 = self.grad_relu(self.h_2)                              # (64, 1024)
        D_2 = self.hessian_relu(self.h_2)                           # (64, 1024)
        h2_hessian = self.recursive_pre_hessian(B_2, W_2, h3_hessian, D_2)  # (64, 1024)
        w2_hessian = h3_hessian.t().matmul(torch.pow(self.x_2, 2))  # (512, 1024)
        # print(B_2.shape, h2_hessian.shape, w2_hessian.shape)

        W_1 = self.weight_fc1                                       # (1024, 2304)
        B_1 = self.grad_relu(self.h_1)                              # (64, 2304)
        D_1 = self.hessian_relu(self.h_1)                           # (64, 2304)
        h1_hessian = self.recursive_pre_hessian(B_1, W_1, h2_hessian, D_1)  # (64, 2304)
        w1_hessian = h2_hessian.t().matmul(torch.pow(self.x_1, 2))  # (1024, 2304)
        # print(B_1.shape, h1_hessian.shape, w1_hessian.shape)

        a = 1

        return w1_hessian, w2_hessian, w3_hessian

    def recursive_pre_hessian(self, B, W, H2, D):
        H1 = torch.pow(B, 2) * H2.matmul(torch.pow(W, 2)) + D
        return H1

    def grad_relu(self, x):
        return (x>0).float()

    def hessian_relu(self, x):
        return x*0


# class AlexNet1(nn.Module):
#     def __init__(self):
#         super(AlexNet1,self).__init__()

#         # 由于MNIST为28x28， 而最初AlexNet的输入图片是227x227的。所以网络层数和参数需要调节
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) #AlexCONV1(3,96, k=11,s=4,p=0)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool1(k=3, s=2)
#         self.relu1 = nn.ReLU()

#         # self.conv2 = nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)#AlexCONV2(96, 256,k=5,s=1,p=2)
#         self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)#AlexPool2(k=3,s=2)
#         self.relu2 = nn.ReLU()


#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)#AlexCONV3(256,384,k=3,s=1,p=1)
#         # self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)#AlexCONV4(384, 384, k=3,s=1,p=1)
#         self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)#AlexCONV5(384, 256, k=3, s=1,p=1)
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)#AlexPool3(k=3,s=2)
#         self.relu3 = nn.ReLU()

#         self.fc6 = nn.Linear(256*3*3, 1024)  #AlexFC6(256*6*6, 4096)
#         self.fc7 = nn.Linear(1024, 512) #AlexFC6(4096,4096)
#         self.fc8 = nn.Linear(512, 10)  #AlexFC6(4096,1000)

#     def forward(self,x):
#         x = self.conv1(x)
#         print(x.shape)
#         x = self.pool1(x)
#         print(x.shape)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         print(x.shape)
#         x = self.pool2(x)
#         print(x.shape)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         print(x.shape)
#         x = self.conv4(x)
#         print(x.shape)
#         x = self.conv5(x)
#         print(x.shape)
#         x = self.pool3(x)
#         print(x.shape)
#         x = self.relu3(x)
#         x = x.view(-1, 256 * 3 * 3)#Alex: x = x.view(-1, 256*6*6)
#         x = self.fc6(x)
#         x = F.relu(x)
#         x = self.fc7(x)
#         x = F.relu(x)
#         x = self.fc8(x)
#         return x


# x = torch.rand(64,1,28,28)
# net = AlexNet()
# x = net(x)
# print(x.shape)
# net.compute_hessian()
# for name, param in net.named_parameters():
# 	print(name, '      ', param.size())