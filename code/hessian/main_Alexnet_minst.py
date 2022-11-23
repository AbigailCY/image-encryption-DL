import os
import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from utils import *
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        fc1, fc2, fc3 = model.compute_hessian()

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))
    return fc1, fc2, fc3

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)

        test_loss += criterion(output, target)
        pred = output.max(1, keepdim=True)[1]                           #
        correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

def plot(epoch, eigenvalues, num_bins=1000, sigma_squared=1e-5):

    e_max = np.max(eigenvalues)
    e_min = np.min(eigenvalues)
    overhead = (np.max(eigenvalues) - np.min(eigenvalues))/15
    lambda_max = e_max + overhead
    lambda_min = e_min - overhead
    print(e_max, e_min)

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * (lambda_max - lambda_min)

    density_output = np.zeros(num_bins)
    
    for j in range(num_bins):
        x = grids[j]
        tmp_result = gaussian(eigenvalues, x, sigma)
        density_output[j] = np.mean(tmp_result)
    density = density_output

    plt.plot(grids, density, linestyle = '-', linewidth=0.5)
    plt.ylabel('Density', fontsize=8, labelpad=6)
    plt.xlabel('Eigenvlaue', fontsize=8, labelpad=6)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.axis([lambda_min-overhead, lambda_max+overhead, None, None])

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batch_size = 64
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    from Alexnet_minst import AlexNet
    model = AlexNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.99))

    toSave3 = np.zeros(50,10,512)
    toSave2 = np.zeros((50,512,1024))
    toSave1 = np.zeros((50,1024,2304))
    toSave4 = np.zeros(50)
    for epoch in range(1, 51):
        print('----------------start train-----------------')
        fc1, fc2, fc3 = train(model, train_loader, optimizer, epoch)
        fc1, fc2, fc3 = fc1.cpu().detach().numpy(), fc2.cpu().detach().numpy(), fc3.cpu().detach().numpy()
        print('----------------end train-----------------')
        print('----------------start test-----------------')
        acc = test(model, test_loader)
        print('----------------end test-----------------')
        toSave1[epoch-1,:,:] = fc1
        toSave2[epoch-1,:,:] = fc2
        toSave3[epoch-1,:,:] = fc3
        toSave4[epoch-1] = acc
    np.save("./hessian_eigen/AM_fc1.npy", toSave1)
    np.save("./hessian_eigen/AM_fc2.npy", toSave2)
    np.save("./hessian_eigen/AM_fc3.npy", toSave3)
    np.save("./hessian_eigen/AM_acc.npy", toSave4)
    # with PdfPages('./cifar_alexnet.pdf') as pdf: 
    #     d = pdf.infodict()
    #     d['Title'] = 'Hessian ESD: Alexnet FC on Cifar'
    #     for epoch in range(1, 101):
    #         print('----------------start train-----------------')
    #         fc1, fc2, fc3 = train(model, train_loader, optimizer, epoch)
    #         print('----------------end train-----------------')

    #         print('----------------start test-----------------')
    #         test(model, test_loader)
    #         print('----------------end test-----------------')

    #         plt.figure(figsize=(11,3))

    #         plt.subplot(131)
    #         plot(epoch, fc1.cpu().detach().numpy())
    #         plt.subplot(131).set_title('Eopch: '+ str(epoch)+" Layer FC1")

    #         plt.subplot(132)
    #         plot(epoch, fc2.cpu().detach().numpy())
    #         plt.subplot(132).set_title('Eopch: '+ str(epoch)+" Layer FC2")

    #         plt.subplot(133)
    #         plot(epoch, fc3.cpu().detach().numpy())
    #         plt.subplot(133).set_title('Eopch: '+ str(epoch)+" Layer FC3")

    #         plt.tight_layout()
    #         pdf.savefig()
    #         plt.close()
        
    torch.save(model.state_dict(), "./checkpoint/alexnet_minst.pkl")
