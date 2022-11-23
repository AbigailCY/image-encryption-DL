import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ComplexNN import *


def encode(a, b, theta):
    x = a * torch.cos(theta) - b * torch.sin(theta)
    y = a * torch.sin(theta) + b * torch.cos(theta)
    return x, y

def decode(a, b, theta):
    return a * torch.cos(-theta) - b * torch.sin(-theta)

def ED(a, b, theta):
    x, y = encode(a, b, theta)
    return decode(x, y, theta)

