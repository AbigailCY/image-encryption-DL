import torch
from torch.nn import functional as F
from models.hessian_cal3 import *


class LeNet_hessian(Module_hessian):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = Id_Conv(1, 6, kernel_size=5, stride=1, padding=0, bias=False)
        self.layer2 = Max_Relu_Conv(6, 16, kernel_size=5, stride=1, padding=0, bias=False,
                                    kernel_size_p=2, stride_p=2, padding_p=0)
        self.layer3 = Max_Relu_FC(16*4*4, 120, bias=False,
                                  kernel_size_p=2, stride_p=2, padding_p=0)
        self.layer4 = Relu_FC(120, 84, bias=False)
        self.layer5 = Relu_FC(84, num_classes, bias=False)

        self.weight_init()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        self.logit = self.layer5(x)

        return self.logit

    def compute_hessian(self, h_hessian_n):
        bs = self.logit.shape[0]
        s = F.softmax(self.logit, dim=1)

        h_hessian = -s.view(bs,10,1).matmul(s.view(bs,1,10))
        d = s * (1 - s)
        for i in range(10):
            h_hessian[:,i,i] = d[:,i]
        h_hessian = torch.symeig(h_hessian)[0]

        h_hessian = h_hessian.view(bs, -1)
        h_hessian = self.layer5.compute_hessian(h_hessian)
        h_hessian = self.layer4.compute_hessian(h_hessian)
        h_hessian = self.layer3.compute_hessian(h_hessian)
        h_hessian = self.layer2.compute_hessian(h_hessian)
        h_hessian = self.layer1.compute_hessian(h_hessian)

        return h_hessian
