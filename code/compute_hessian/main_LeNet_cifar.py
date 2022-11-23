import os
import torch
from torch import nn
# from LeNet_cifar import LeNet
from torchvision import datasets, transforms
import time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from utils import *

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # fc3, fc2, fc1, conv2, conv1 = model.compute_hessian(data)

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))
    # return fc3, fc2, fc1, conv2, conv1

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        if batch_idx == 1:
            fc3, fc2, fc1, conv2, conv1 = model.compute_hessian(data)

        test_loss += criterion(output, target)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))
    
    return fc3, fc2, fc1, conv2, conv1


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    batch_size = 63
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    train_dataset = datasets.CIFAR10(root='../', train=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    from LeNet_cifar import LeNet

    model = LeNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

    # t = time.time()
    with PdfPages('./graph/'+'lenet_cifar'+'_hist.pdf') as pdf: 
        for epoch in range(1, 5):
            print('----------------start train-----------------')
            train(model, train_loader, optimizer, epoch)
            print('----------------end train-----------------')

            print('----------------start test-----------------')
            fc3, fc2, fc1, conv2, conv1 = test(model, test_loader)
            print('----------------end test-----------------')

            fc3, fc2, fc1 = fc3.cpu().detach().numpy().reshape(-1), fc2.cpu().detach().numpy().reshape(-1), fc1.cpu().detach().numpy().reshape(-1)
            conv2, conv1 = conv2.cpu().detach().numpy().reshape(-1), conv1.cpu().detach().numpy().reshape(-1)


            plt.figure(figsize=(24, 4))
            plt.subplot(151)
            weights = np.ones_like(fc3)/float(len(fc3))
            plt.hist(fc3, bins = 100, weights=weights)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplot(151).set_title('Eopch: '+ str(epoch)+' Layer FC3')

            plt.subplot(152)
            weights = np.ones_like(fc2)/float(len(fc2))
            plt.hist(fc2, bins = 100, weights=weights)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplot(152).set_title('Eopch: '+ str(epoch)+' Layer FC2')

            plt.subplot(153)
            weights = np.ones_like(fc1)/float(len(fc1))
            plt.hist(fc1, bins = 100, weights=weights)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplot(153).set_title('Eopch: '+ str(epoch)+' Layer FC1')

            plt.subplot(154)
            weights = np.ones_like(conv2)/float(len(conv2))
            plt.hist(conv2, bins = 100, weights=weights)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplot(154).set_title('Eopch: '+ str(epoch)+' Layer conv2')

            plt.subplot(155)
            weights = np.ones_like(conv1)/float(len(conv1))
            plt.hist(conv1, bins = 100, weights=weights)
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.subplot(155).set_title('Eopch: '+ str(epoch)+' Layer conv1')

            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print('Epoch ' + str(epoch) + ' finished')


    # print('time:', time.time()-t)
