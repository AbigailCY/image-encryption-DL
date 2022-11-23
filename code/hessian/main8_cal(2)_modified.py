import os
import torch
from torch import nn
from LeNet8 import LeNet
from torchvision import datasets, transforms
import time
import numpy as np


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        ## -------------
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        xs = optimizer.param_groups[0]['params']
        ys = loss  # put your own loss into ys
        grads = torch.autograd.grad(ys, xs, create_graph=True)  # first order gradient
        grads2 = get_second_order_grad(grads, xs)  # second order gradient
        ## -------------

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def get_second_order_grad(grads, xs):
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        start = time.time()
        print('2nd order on ', j, 'th layer')
        grad = torch.reshape(grad, [-1])
        grads2_tmp = torch.zeros(len(grad), len(grad)).cuda()
        for i, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp[i, :] = g2
        grads2.append(grads2_tmp)
        print('Time used is ', time.time() - start)

    return grads2


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)

        test_loss += criterion(output, target)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    batch_size = 63
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = LeNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

    t = time.time()
    for epoch in range(1, 5):
        print('----------------start train-----------------')
        train(model, train_loader, optimizer, epoch)
        print('----------------end train-----------------')

        print('----------------start test-----------------')
        test(model, test_loader)
        print('----------------end test-----------------')

    print('time:', time.time()-t)
