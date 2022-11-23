import os
import torch
from torch import nn
from models.resnet_hessian2 import ResNet18_hessian
from torchvision import datasets, transforms
import time
from utils import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        # model.compute_hessian(None)
        # for name, value in model.named_hessians():
        #     print(name, value.shape)
        # for value in model.hessians():
        #     print(value.shape)
        # print(1)
        # for name, value in model.named_parameters():
        #     print(name, value.shape)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch+1, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def evaluate(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            N = len(data)
            data, target = data.cuda(), target.cuda()
            output = model(data)

            test_loss += N * criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))

def plot(model, train_loader, criterion, pdf, epoch):
    hessians = []
    model.eval()
    for inputs, labels in train_loader:
        hessian_data = (inputs, labels)
        inputs = inputs.cuda()
        model(inputs)
        model.compute_hessian(None)
        for value in model.hessians():
            hessians.append(value.cpu().detach().numpy().reshape(-1))
        break
    hessian_comp = hessian(model, criterion, data=hessian_data)
    hessian_comp.separate_plot(pdf, epoch)
    return hessians



def plot_hist(all_hessians):
    epochs = len(all_hessians)
    num = len(all_hessians[0])

    with PdfPages('./graph/plot_FormulaH.pdf') as pdf:

        for epoch in range(epochs):
            fig = plt.figure(figsize=(5*num,4))

            for i in range(num):
                fig.add_subplot(1, num, i+1).set_title('Epoch '+str(epoch+1)+' Layer ' + str(i+1))
                hessian = all_hessians[epoch][i]
                weights = np.ones_like(hessian)/float(len(hessian))
                plt.hist(hessian, bins = 100, weights=weights)


            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print('Epoch ' + str(epoch) + ' finished') 

        for epoch in range(epochs):
            fig = plt.figure(figsize=(5*num,4))

            for i in range(num):
                ax = fig.add_subplot(1, num, i+1)
                plt.title('Epoch '+str(epoch+1)+' Layer ' + str(i+1))
                hessian = all_hessians[epoch][i]
                hessian = np.delete(hessian, np.argwhere(hessian<1e-5))
                weights = np.ones_like(hessian)/float(len(hessian))
                plt.hist(hessian, bins = 3000, weights=weights)
                ax.set_xscale('log')
                plt.xlim(left=1e-5)

            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print('Epoch ' + str(epoch) + ' finished')

            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print('Epoch ' + str(epoch) + ' finished')



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    batch_size = 127
    trans_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.Pad(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trans_test = transforms.Compose([
        # transforms.Pad(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, transform=trans_train)
    test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, transform=trans_test)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNet18_hessian(len(train_dataset.classes)).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.1)
    t = time.time()
    hessians_all = []
    with PdfPages('./graph/plot_EstimateH.pdf') as pdf:
        for epoch in range(1):
            print('----------------start train-----------------')
            train(model, train_loader, optimizer, epoch)
            lr_scheduler.step()
            print('----------------end train-----------------')

            # print('----------------start test-----------------')
            # evaluate(model, test_loader)
            # print('----------------end test-----------------')
            print('time:', time.time()-t)
            t = time.time()

            print('----------------start estimate plot-----------------')
            hessians = plot(model, train_loader, criterion, pdf, epoch+1)
            hessians_all.append(hessians)
            print('time:', time.time()-t)
            t = time.time()
            print('----------------end estimate plot-----------------')


    print('----------------start formula plot-----------------')
    plot_hist(hessians_all)
    print('time:', time.time()-t)
    print('----------------end formula plot-----------------')






