import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from collections import OrderedDict
from itertools import islice
import operator

import torch
from torch._jit_internal import _copy_to_script_wrapper


class Module_hessian(nn.Module):
    def __init__(self):
        super(Module_hessian, self).__init__()
        self._hessians = OrderedDict()
        self.H_shape = None
        self.activation = None

    def compute_hessian(self, h_hessian):
        raise NotImplementedError

    def recursive_pre_hessian(self, W, h, H2):
        if self.activation == 'relu':
            B = (h > 0).float()
            D = h * 0
        elif self.activation == 'identity':
            B = torch.ones_like(h)
            D = h * 0
        elif self.activation == 'sigmoid':
            B = h * (1 - h)
            D = h * (1 - h) * (1 - 2 * h)
        else:
            raise NotImplementedError

        if W == 'identity':
            H1 = B.pow(2) * H2 + D
        else:
            H1 = B.pow(2) * H2.matmul(W.pow(2)) + D

        return H1

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def register_hessian(self, name, H):
        self._hessians[name] = H.view(self.H_shape)

    def hessians(self):
        for name, H in self.named_hessians():
            yield H

    def named_hessians(self, memo=None, prefix=''):
        if memo is None:
            memo = set()

        for name, H in self._hessians.items():
            if H is not None and H not in memo:
                memo.add(H)
                yield prefix + ('.' if prefix else '') + name, H
        for mname, module in self.named_children():
            if isinstance(module, Module_hessian):
                submodule_prefix = prefix + ('.' if prefix else '') + mname
                for name, H in module.named_hessians(memo, submodule_prefix):
                    yield name, H


class Relu_FC(Module_hessian):
    def __init__(self, inplanes, planes, bias):
        super(Relu_FC, self).__init__()

        self.relu = nn.ReLU()
        self.activation = 'relu'
        self.fc = nn.Linear(inplanes, planes, bias=bias)
        self.H_shape = self.fc.weight.shape

        self.weight_init()

    def forward(self, x):
        self.x = x
        self.x_relu = self.relu(self.x)
        self.h = self.fc(self.x_relu)

        return self.h

    def compute_hessian(self, h_hessian):
        W = self.fc.weight
        h = self.x
        x = self.x_relu
        hessian = h_hessian.t().matmul(x.pow(2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian)
        self.register_hessian('fc.weight.hessian', hessian)

        return h_hessian


class BN_Relu_Conv(Module_hessian):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias):
        super(BN_Relu_Conv, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU()
        self.activation = 'relu'
        self.conv = nn.Conv2d(self.inplanes, self.planes, kernel_size, stride=stride, padding=padding, bias=bias)
        self.H_shape = self.conv.weight.shape

        self.weight_init()

    def forward(self, x):
        self.x = x
        self.x_bn = self.bn(self.x)
        x_relu = self.relu(self.x_bn)
        x_conv = self.conv(x_relu)

        return x_conv

    def compute_hessian(self, h_hessian):
        bs = h_hessian.shape[0]
        x_unfold = F.unfold(self.x_bn, self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_relu = F.relu(x_unfold)

        W = self.conv.weight
        c_out, c_in, kh, kw = W.shape
        W = W.view(c_out, c_in * kh * kw)
        h_hessian = h_hessian.view(bs, c_out, -1).transpose(1, 2).reshape(-1, c_out)
        h = x_unfold.reshape(-1, c_in * kh * kw)
        x = x_relu.reshape(-1, c_in * kh * kw)
        hessian = h_hessian.t().matmul(x.pow(2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian)
        self.register_hessian('conv.weight.hessian', hessian)

        output_size = self.x_bn.shape[2:4]
        kernel_size = (kh, kw)
        h_hessian = h_hessian.view(bs, -1, c_in * kh * kw).transpose(1, 2)
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=self.stride, padding=self.padding)

        sigma2 = self.x.std((0, 2, 3), False, keepdim=True)
        gamma = self.bn.weight.view(1, -1, 1, 1)
        h_hessian = h_hessian * gamma.pow(2) / (sigma2 + self.bn.eps)

        return h_hessian


class BN_Max_Relu_Conv(Module_hessian):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias, kernel_size_p, stride_p, padding_p):
        super(BN_Max_Relu_Conv, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size_p = _pair(kernel_size_p)
        self.stride_p = stride_p
        self.padding_p = padding_p

        self.bn = nn.BatchNorm2d(self.inplanes)
        self.pool = nn.MaxPool2d(kernel_size_p, stride_p, padding_p, return_indices=True)
        self.relu = nn.ReLU()
        self.activation = 'relu'
        self.conv = nn.Conv2d(self.inplanes, self.planes, kernel_size, stride=stride, padding=padding, bias=bias)
        self.H_shape = self.conv.weight.shape

        self.weight_init()

    def forward(self, x):
        self.x = x
        self.x_bn = self.bn(self.x)
        self.x_pool, self.inds_pool = self.pool(self.x_bn)
        x_relu = self.relu(self.x_pool)
        x_conv = self.conv(x_relu)

        return x_conv

    def compute_hessian(self, h_hessian):
        bs = h_hessian.shape[0]
        x_unfold = F.unfold(self.x_pool, self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_relu = F.relu(x_unfold)

        W = self.conv.weight
        c_out, c_in, kh, kw = W.shape
        W = W.view(c_out, c_in * kh * kw)
        h_hessian = h_hessian.view(bs, c_out, -1).transpose(1, 2).reshape(-1, c_out)
        h = x_unfold.reshape(-1, c_in * kh * kw)
        x = x_relu.reshape(-1, c_in * kh * kw)
        hessian = h_hessian.t().matmul(x.pow(2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian)
        self.register_hessian('conv.weight.hessian', hessian)

        output_size = self.x_bn.shape[2:4]
        kernel_size = (kh, kw)
        h_hessian = h_hessian.view(bs, -1, c_in * kh * kw).transpose(1, 2)
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=self.stride, padding=self.padding)

        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool, kernel_size=self.kernel_size_p,
                                   stride=self.stride_p, padding=self.padding_p, output_size=self.x.shape[2:4])

        sigma2 = self.x.std((0, 2, 3), False, keepdim=True)
        gamma = self.bn.weight.view(1, -1, 1, 1)
        h_hessian = h_hessian * gamma.pow(2) / (sigma2 + self.bn.eps)

        return h_hessian


class BN_hessian(Module_hessian):
    def __init__(self, planes):
        super(BN_hessian, self).__init__()

        self.bn = nn.BatchNorm2d(planes)
        self.weight_init()

    def forward(self, x):
        self.x = x
        x_bn = self.bn(self.x)

        return x_bn

    def compute_hessian(self, h_hessian):
        sigma2 = self.x.std((0, 2, 3), False, keepdim=True)
        gamma = self.bn.weight.view(1, -1, 1, 1)
        h_hessian = h_hessian * gamma.pow(2) / (sigma2 + self.bn.eps)

        return h_hessian


class Relu_Conv(Module_hessian):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias):
        super(Relu_Conv, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        self.relu = nn.ReLU()
        self.activation = 'relu'
        self.conv = nn.Conv2d(self.inplanes, self.planes, kernel_size, stride=stride, padding=padding, bias=bias)
        self.H_shape = self.conv.weight.shape

        self.weight_init()

    def forward(self, x):
        self.x = x
        x_relu = self.relu(self.x)
        x_conv = self.conv(x_relu)

        return x_conv

    def compute_hessian(self, h_hessian):
        bs = h_hessian.shape[0]
        x_unfold = F.unfold(self.x, self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_relu = F.relu(x_unfold)

        W = self.conv.weight
        c_out, c_in, kh, kw = W.shape
        W = W.view(c_out, c_in * kh * kw)
        h_hessian = h_hessian.view(bs, c_out, -1).transpose(1, 2).reshape(-1, c_out)
        h = x_unfold.reshape(-1, c_in * kh * kw)
        x = x_relu.reshape(-1, c_in * kh * kw)
        hessian = h_hessian.t().matmul(x.pow(2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian)
        self.register_hessian('conv.weight.hessian', hessian)

        output_size = self.x.shape[2:4]
        kernel_size = (kh, kw)
        h_hessian = h_hessian.view(bs, -1, c_in * kh * kw).transpose(1, 2)
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=self.stride, padding=self.padding)

        return h_hessian


class Relu_hessian(Module_hessian):
    def __init__(self):
        super(Relu_hessian, self).__init__()

        self.activation = 'relu'

    def forward(self, x):
        self.x = x
        x_relu = F.relu(self.x)

        return x_relu

    def compute_hessian(self, h_hessian):
        bs = h_hessian.shape[0]
        c_out = c_in = h_hessian.shape[1]
        h_hessian = h_hessian.view(bs, c_out, -1).transpose(1, 2).reshape(-1, c_out)
        h = self.x.view(-1, c_in)
        h_hessian = self.recursive_pre_hessian('identity', h, h_hessian)

        output_size = self.x.shape[2:4]
        h_hessian = h_hessian.view(bs, -1, c_in).transpose(1, 2)
        h_hessian = F.fold(h_hessian, output_size, (1, 1), stride=(1, 1), padding=(0, 0))

        return h_hessian


class Id_Conv(Module_hessian):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias):
        super(Id_Conv, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        self.activation = 'identity'
        self.conv = nn.Conv2d(self.inplanes, self.planes, kernel_size, stride=stride, padding=padding, bias=bias)
        self.H_shape = self.conv.weight.shape

        self.weight_init()

    def forward(self, x):
        self.x = x
        x_conv = self.conv(self.x)

        return x_conv

    def compute_hessian(self, h_hessian):
        bs = h_hessian.shape[0]
        x_unfold = F.unfold(self.x, self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_relu = x_unfold

        W = self.conv.weight
        c_out, c_in, kh, kw = W.shape
        W = W.view(c_out, c_in * kh * kw)
        h_hessian = h_hessian.view(bs, c_out, -1).transpose(1, 2).reshape(-1, c_out)
        h = x_unfold.reshape(-1, c_in * kh * kw)
        x = x_relu.reshape(-1, c_in * kh * kw)
        hessian = h_hessian.t().matmul(x.pow(2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian)
        self.register_hessian('conv.weight.hessian', hessian)

        output_size = self.x.shape[2:4]
        kernel_size = (kh, kw)
        h_hessian = h_hessian.view(bs, -1, c_in * kh * kw).transpose(1, 2)
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=self.stride, padding=self.padding)

        return h_hessian


class Max_Relu_Conv(Module_hessian):
    def __init__(self, inplanes, planes, kernel_size, stride, padding, bias, kernel_size_p, stride_p, padding_p):
        super(Max_Relu_Conv, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.kernel_size_p = _pair(kernel_size_p)
        self.stride_p = stride_p
        self.padding_p = padding_p

        self.pool = nn.MaxPool2d(kernel_size_p, stride_p, padding_p, return_indices=True)
        self.relu = nn.ReLU()
        self.activation = 'relu'
        self.conv = nn.Conv2d(self.inplanes, self.planes, kernel_size, stride=stride, padding=padding, bias=bias)
        self.H_shape = self.conv.weight.shape

        self.weight_init()

    def forward(self, x):
        self.x = x
        self.x_pool, self.inds_pool = self.pool(self.x)
        x_relu = self.relu(self.x_pool)
        x_conv = self.conv(x_relu)

        return x_conv

    def compute_hessian(self, h_hessian):
        bs = h_hessian.shape[0]
        x_unfold = F.unfold(self.x_pool, self.kernel_size, stride=self.stride, padding=self.padding)
        x_unfold = x_unfold.transpose(1, 2)
        x_relu = F.relu(x_unfold)

        W = self.conv.weight
        c_out, c_in, kh, kw = W.shape
        W = W.view(c_out, c_in * kh * kw)
        h_hessian = h_hessian.view(bs, c_out, -1).transpose(1, 2).reshape(-1, c_out)
        h = x_unfold.reshape(-1, c_in * kh * kw)
        x = x_relu.reshape(-1, c_in * kh * kw)
        hessian = h_hessian.t().matmul(x.pow(2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian)
        self.register_hessian('conv.weight.hessian', hessian)

        output_size = self.x_pool.shape[2:4]
        kernel_size = (kh, kw)
        h_hessian = h_hessian.view(bs, -1, c_in * kh * kw).transpose(1, 2)
        h_hessian = F.fold(h_hessian, output_size, kernel_size, stride=self.stride, padding=self.padding)

        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool, kernel_size=self.kernel_size_p,
                                   stride=self.stride_p, padding=self.padding_p, output_size=self.x.shape[2:4])

        return h_hessian


class Max_Relu_FC(Module_hessian):
    def __init__(self, inplanes, planes, bias, kernel_size_p, stride_p, padding_p):
        super(Max_Relu_FC, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size_p, stride_p, padding_p, return_indices=True)
        self.relu = nn.ReLU()
        self.activation = 'relu'
        self.fc = nn.Linear(inplanes, planes, bias=bias)
        self.H_shape = self.fc.weight.shape

        self.weight_init()

    def forward(self, x):
        self.x = x
        self.x_pool, self.inds_pool = self.pool(self.x)
        self.x_relu = self.relu(self.x_pool)
        self.h = self.fc(self.x_relu)

        return self.h

    def compute_hessian(self, h_hessian):
        W = self.fc.weight
        h = self.h
        x = self.x_relu
        hessian = h_hessian.t().matmul(x.pow(2))
        h_hessian = self.recursive_pre_hessian(W, h, h_hessian)
        self.register_hessian('conv.weight.hessian', hessian)

        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool, kernel_size=self.kernel_size_p,
                                   stride=self.stride_p, padding=self.padding_p, output_size=self.x.shape[2:4])

        return h_hessian


class Max_Relu(Module_hessian):
    def __init__(self, kernel_size_p, stride_p, padding_p):
        super(Max_Relu, self).__init__()

        self.kernel_size_p = _pair(kernel_size_p)
        self.stride_p = stride_p
        self.padding_p = padding_p

        self.activation = 'relu'

    def forward(self, x):
        self.x = x
        self.x_pool, self.inds_pool = F.max_pool2d_with_indices(self.x, self.kernel_size_p, stride=self.stride_p,
                                                                padding=self.padding_p)
        x_relu = F.relu(self.x_pool)

        return x_relu

    def compute_hessian(self, h_hessian):
        bs = h_hessian.shape[0]
        c_out = c_in = h_hessian.shape[1]
        h_hessian = h_hessian.view(bs, c_out, -1).transpose(1, 2).reshape(-1, c_out)
        h = self.x_pool.view(-1, c_in)
        h_hessian = self.recursive_pre_hessian('identity', h, h_hessian)

        output_size = self.x_pool.shape[2:4]
        h_hessian = h_hessian.view(bs, -1, c_in).transpose(1, 2)
        h_hessian = F.fold(h_hessian, output_size, (1, 1), stride=(1, 1), padding=(0, 0))

        h_hessian = F.max_unpool2d(h_hessian, self.inds_pool, kernel_size=self.kernel_size_p,
                                   stride=self.stride_p, padding=self.padding_p, output_size=self.x.shape[2:4])

        return h_hessian


class Sequential_hessian(Module_hessian):
    def __init__(self, *args):
        super(Sequential_hessian, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    @_copy_to_script_wrapper
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    @_copy_to_script_wrapper
    def __len__(self):
        return len(self._modules)

    @_copy_to_script_wrapper
    def __iter__(self):
        return iter(self._modules.values())

    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def compute_hessian(self, h_hessian):
        for i in range(len(self) - 1, -1, -1):
            h_hessian = self._modules[str(i)].compute_hessian(h_hessian)
        return h_hessian
