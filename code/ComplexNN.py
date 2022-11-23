import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    def forward(self, x1, x2):
        c1 = (x1*x1 + x2*x2).sqrt()
        c = torch.mean(c1,(0))
        non_Linear = c1 / torch.max(c1, c)
        return torch.mul(x1, non_Linear), torch.mul(x2, non_Linear)

class _ComplexBatchNorm2(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm2, self).__init__()
        self.num_features = num_features
        # self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features,3))
            self.bias = nn.Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features,2))
            self.register_buffer('running_covar', torch.zeros(num_features,3))
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:,0] = 1.4142135623730951
            self.running_covar[:,1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:,:2],1.4142135623730951)
            nn.init.zeros_(self.weight[:,2])
            nn.init.zeros_(self.bias)
            
class ComplexBatchNorm2d2(_ComplexBatchNorm2):

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum


        if self.training:

            # calculate mean of real and imaginary part
            mean_r = input_r.mean([0, 2, 3])
            mean_i = input_i.mean([0, 2, 3])


            mean = torch.stack((mean_r,mean_i),dim=1)

            # update running mean
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean

            input_r = input_r-mean_r[None, :, None, None]
            input_i = input_i-mean_i[None, :, None, None]

            # Elements of the covariance matrix (biased for train)
            n = input_r.numel() / input_r.size(1)
            Crr = 1./n*input_r.pow(2).sum(dim=[0,2,3]) #+self.eps
            Cii = 1./n*input_i.pow(2).sum(dim=[0,2,3]) #+self.eps
            Cri = (input_r.mul(input_i)).mean(dim=[0,2,3])

            with torch.no_grad():
                self.running_covar[:,0] = exponential_average_factor * Crr * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,0]

                self.running_covar[:,1] = exponential_average_factor * Cii * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_covar[:,1]

                # self.running_covar[:,2] = exponential_average_factor * Cri * n / (n - 1)\
                #     + (1 - exponential_average_factor) * self.running_covar[:,2]

        else:
            mean = self.running_mean
            Crr = self.running_covar[:,0] #+self.eps
            Cii = self.running_covar[:,1] #+self.eps
            # Cri = self.running_covar[:,2] #+self.eps

            input_r = input_r-mean[None,:,0,None,None]
            input_i = input_i-mean[None,:,1,None,None]

        # calculate the inverse square root the covariance matrix
        inverse_st = torch.sqrt(Crr + Cii)
        # det = Crr*Cii-Cri.pow(2)
        # s = torch.sqrt(det)
        # t = torch.sqrt(Cii+Crr + 2 * s)
        # inverse_st = 1.0 / (s * t)
        # Rrr = (Cii + s) * inverse_st
        # Rii = (Crr + s) * inverse_st
        # Rri = -Cri * inverse_st
        # input_r, input_i = Rrr[None,:,None,None]*input_r+Rri[None,:,None,None]*input_i, \
        #                    Rii[None,:,None,None]*input_i+Rri[None,:,None,None]*input_r
        input_r, input_i = inverse_st[None,:,None,None]*input_r, inverse_st[None,:,None,None]*input_i
                           

        if self.affine:
            input_r, input_i = self.weight[None,:,0,None,None]*input_r+self.weight[None,:,2,None,None]*input_i+\
                               self.bias[None,:,0,None,None], \
                               self.weight[None,:,2,None,None]*input_r+self.weight[None,:,1,None,None]*input_i+\
                               self.bias[None,:,1,None,None]

        return input_r, input_i

class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, 
                 count_include_pad=True, divisor_override=None):
        super(ComplexAvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x1, x2):
        return F.avg_pool2d(x1, self.kernel_size, self.stride, self.padding, self.ceil_mode, 
                self.count_include_pad, self.divisor_override), \
               F.avg_pool2d(x2, self.kernel_size, self.stride, self.padding, self.ceil_mode, 
                self.count_include_pad, self.divisor_override)

class ComplexMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None,
                            padding=0, dilation=1, return_indices = False, ceil_mode=False):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.dilation = dilation
        # self.return_indices = return_indices

    def forward(self, x1, x2):
        x_norm = x1*x1 + x2*x2
        x_norm_max, inds = F.max_pool2d(x_norm, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, True)
        n,c,h1,w1 = x_norm.shape
        n,c,h2,w2 = x_norm_max.shape
        X1 = x1.view(n,c,h1*w1)
        X2 = x2.view(n,c,h1*w1)
        inds_X = inds.view(n,c,h2*w2)
        X1_max = X1.gather(2, inds_X).view(n,c,h2,w2)
        X2_max = X2.gather(2, inds_X).view(n,c,h2,w2)

        return X1_max, X2_max

class ComplexDropout(nn.Module):
    def __init__(self, p):
        super(ComplexDropout, self).__init__()
        self.p = p
    def forward(self, x1, x2):
        sample = torch.ones_like(x1)* self.p
        S = torch.distributions.Bernoulli(sample).sample()/(1-self.p)
        return x1 * S, x2 * S

def encode(a, b, theta):
    x = a * torch.cos(theta) - b * torch.sin(theta)
    y = a * torch.sin(theta) + b * torch.cos(theta)
    return x, y

def decode(x, y, theta):
    return x * torch.cos(-theta) - y * torch.sin(-theta)

def ED(a, b, theta):
    return a * torch.cos(theta) - b * torch.sin(theta)





class _ComplexBatchNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        # self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features,2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_covar', torch.zeros(num_features))
            self.running_covar[:] = 1.4142135623730951
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_covar.zero_()
            self.running_covar[:] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:],1.4142135623730951)
            nn.init.zeros_(self.bias)


class ComplexBatchNorm2d(_ComplexBatchNorm):

    def forward(self, input_r, input_i):
        assert(input_r.size() == input_i.size())
        assert(len(input_r.shape) == 4)
        exponential_average_factor = 0.0


        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum


        if self.training:

            vsum = input_r*input_r + input_i*input_i
            var = vsum.mean([0, 2, 3])
            with torch.no_grad():
                self.running_covar = exponential_average_factor * var\
                    + (1 - exponential_average_factor) * self.running_covar

        else:
            var = self.running_covar

        # calculate the inverse square root the covariance matrix
        inverse_st = 1/torch.sqrt(var)

        input_r, input_i = inverse_st[None,:,None,None]*input_r, inverse_st[None,:,None,None]*input_i
                           

        if self.affine:
            input_r, input_i = self.weight[None,:,None,None]*input_r+\
                               self.bias[None,:,0,None,None], \
                               self.weight[None,:,None,None]*input_i+\
                               self.bias[None,:,1,None,None]

        return input_r, input_i


