import torch
from torch.nn import functional as F
from models.hessian_cal3 import *


class BasicBlock(Module_hessian):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=False, pre_max_pool=False):
        super(BasicBlock, self).__init__()

        self.downsample = downsample
        self.pre_max_pool = pre_max_pool
        self.stride = stride

        if pre_max_pool:
            self.layer1 = Max_Relu_Conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False,
                                        kernel_size_p=3, stride_p=2, padding_p=1)
        else:
            self.layer1 = Relu_Conv(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer2 = BN_Relu_Conv(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3 = BN_hessian(planes)

        if pre_max_pool:
            self.layer4 = Max_Relu(3, stride_p=2, padding_p=1)
        elif downsample and not pre_max_pool:
            self.layer4 = Sequential_hessian(
                Relu_Conv(inplanes, planes * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                BN_hessian(planes * self.expansion))
        else:
            self.layer4 = Relu_hessian()

    def forward(self, x):
        self.x = x

        identity = self.layer4(self.x)
        x1 = self.layer1(self.x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_sum = x3 + identity

        return x_sum

    def compute_hessian(self, h_hessian):
        h_hessian_x3 = h_hessian
        h_hessian_id = h_hessian

        h_hessian_x2 = self.layer3.compute_hessian(h_hessian_x3)
        h_hessian_x1 = self.layer2.compute_hessian(h_hessian_x2)
        h_hessian_x = self.layer1.compute_hessian(h_hessian_x1)
        h_hessian_x_copy = self.layer4.compute_hessian(h_hessian_id)

        h_hessian_x += h_hessian_x_copy

        return h_hessian_x


class Bottleneck(Module_hessian):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=False, pre_max_pool=False):
        super(Bottleneck, self).__init__()

        self.downsample = downsample
        self.pre_max_pool = pre_max_pool
        self.stride = stride

        if pre_max_pool:
            self.layer1 = Max_Relu_Conv(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False,
                                        kernel_size_p=3, stride_p=2, padding_p=1)
        else:
            self.layer1 = Relu_Conv(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer2 = BN_Relu_Conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.layer3 = BN_Relu_Conv(planes, planes*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.layer4 = BN_hessian(planes*self.expansion)

        if pre_max_pool:
            self.layer5 = Max_Relu(3, stride_p=2, padding_p=1)
        elif downsample and not pre_max_pool:
            self.layer5 = Sequential_hessian(
                Relu_Conv(inplanes, planes * self.expansion, kernel_size=1, stride=stride, padding=0, bias=False),
                BN_hessian(planes * self.expansion))
        else:
            self.layer5 = Relu_hessian()

    def forward(self, x):
        self.x = x

        identity = self.layer5(self.x)
        x1 = self.layer1(self.x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x_sum = x4 + identity

        return x_sum

    def compute_hessian(self, h_hessian):
        h_hessian_x4 = h_hessian
        h_hessian_id = h_hessian

        h_hessian_x3 = self.layer4.compute_hessian(h_hessian_x4)
        h_hessian_x2 = self.layer3.compute_hessian(h_hessian_x3)
        h_hessian_x1 = self.layer2.compute_hessian(h_hessian_x2)
        h_hessian_x = self.layer1.compute_hessian(h_hessian_x1)
        h_hessian_x_copy = self.layer5.compute_hessian(h_hessian_id)

        h_hessian_x += h_hessian_x_copy

        return h_hessian_x


class layer0(Module_hessian):
    def __init__(self, inplanes):
        super(layer0, self).__init__()

        self.inplanes = inplanes
        self.conv = Id_Conv(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = BN_hessian(self.inplanes)

    def forward(self, x):
        self.x = x
        x_conv = self.conv(self.x)
        x_bn = self.bn(x_conv)

        return x_bn

    def compute_hessian(self, h_hessian):
        h_hessian = self.bn.compute_hessian(h_hessian)
        h_hessian = self.conv.compute_hessian(h_hessian)

        return h_hessian


class ResNet18_hessian(Module_hessian):
    def __init__(self, num_classes=1000, pretrain=False):
        super(ResNet18_hessian, self).__init__()
        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.inplanes = 64
        self.layer0 = layer0(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, max_pool=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Relu_FC(512 * block.expansion, num_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if pretrain:
            state_dict = torch.load('model', map_location='cpu')
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')

            state_dict = {k: v for k, v in self.state_dict().items() if k in self.state_dict()}

            model_state_dict = self.state_dict()
            model_state_dict.update(state_dict)

            self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1, max_pool=False):
        downsample = False
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = True

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, pre_max_pool=max_pool))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential_hessian(*layers)

    def forward(self, x):
        x = self.layer0(x) # (-,64,16,16)
        x = self.layer1(x) # (-,64,8,8)
        x = self.layer2(x) # (-,128,4,4)
        x = self.layer3(x) # (-,256,2,2)
        x = self.layer4(x) # (-,512,1,1)

        self.final_pool_size = x.shape[2:]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        self.logit = self.fc(x)

        return self.logit

    def compute_hessian(self, h_hessian_n):
        bs = self.logit.shape[0]
        s = F.softmax(self.logit, dim=1)

        h_hessian = -s.view(bs, -1, 1).matmul(s.view(bs, 1, -1))
        d = s * (1 - s)
        for i in range(10):
            h_hessian[:,i,i] = d[:,i]
        h_hessian = torch.symeig(h_hessian)[0]

        h_hessian = h_hessian.view(bs, -1)
        h_hessian = self.fc.compute_hessian(h_hessian)

        h_hessian = h_hessian.view(bs, -1, 1, 1)
        kh, kw = self.final_pool_size
        size_h = kh * h_hessian.shape[2]
        size_w = kw * h_hessian.shape[3]
        h_hessian = F.interpolate(h_hessian, (size_h, size_w), mode='area') / (kh * kw)

        h_hessian = self.layer4.compute_hessian(h_hessian) # (-,256,2,2)
        h_hessian = self.layer3.compute_hessian(h_hessian) # (-,128,4,4)
        h_hessian = self.layer2.compute_hessian(h_hessian) # (-,64,8,8)
        h_hessian = self.layer1.compute_hessian(h_hessian) # (-,64,16,16)
        h_hessian = self.layer0.compute_hessian(h_hessian) # (-,3,32,32)

        return h_hessian
