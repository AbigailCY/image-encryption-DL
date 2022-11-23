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
        # if batch_idx == 0:
        #     for name, value in model.named_hessians():
        #         print(name, value.shape)
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
    model.eval()
    for inputs, labels in train_loader:
        hessian_data = (inputs, labels)
        break
    hessian_comp = hessian(model, criterion, data=hessian_data)
    hessian_comp.separate_plot(pdf, epoch)

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
    with PdfPages('./graph/plot_pyhessian.pdf') as pdf:
        for epoch in range(50):
            print('----------------start train-----------------')
            train(model, train_loader, optimizer, epoch)
            lr_scheduler.step()
            print('----------------end train-----------------')

            print('----------------start test-----------------')
            evaluate(model, test_loader)
            print('----------------end test-----------------')
            print('----------------start plot-----------------')
            plot(model, train_loader, criterion, pdf, epoch+1)
            print('----------------end plot-----------------')
    print('time:', time.time()-t)

    

    # params = []
    # grads = []
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:
    #         continue
    #     if name.find('conv.weight') != -1 or name.find('fc.weight') != -1:
    #         params.append(param)
    #         grads.append(0. if param.grad is None else param.grad + 0.)
    #         print(name, param.shape)
    # print(len(params), len(grads))
    
        # for epoch in range(1,2):
        #     plot(model, train_loader, criterion, pdf, epoch)
    





    

 
    # model.eval()
    # for inputs, labels in train_loader:
    #     hessian_data = (inputs, labels)
    #     break
    # print('time:', time.time()-t)
    
    # hessian_comp = hessian(model, criterion, dataloader=train_loader)
    # # hessian_comp = hessian(model, criterion, data=hessian_data)
    # print('time:', time.time()-t)
    # density_eigen, density_weight = hessian_comp.density()
    # eigenvalues = np.array(density_eigen)
    # weight = np.array(density_weight)
    # print(eigenvalues.shape, weight.shape)

    # print('time:', time.time()-t)

    # get_esd_plot(density_eigen, density_weight)






