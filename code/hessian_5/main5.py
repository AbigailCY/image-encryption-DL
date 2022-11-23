import os
import numpy as np
import torch
from torch import nn
from models.resnet_hessian3 import ResNet18_hessian
from models.custom_loss import corr_regularizer
from torchvision import datasets, transforms
import time
from utils_h import *


def train(model, train_loader, optimizer, regularizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        reg = regularizer.forward(model, 1e-3)
        if epoch>5:
            loss_total = loss + reg
        else:
            loss_total = loss
        loss_total.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f} Reg: {:.5f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), reg.item()))


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


def compute_hessain_value(model, criterion, hessian_data):
    hessian_comp = hessian_class(model, criterion, data=hessian_data)
    all_eigen = hessian_comp.eigendata()  # [num_layers,100]

    num_layers = all_eigen.shape[0]
    hessians = np.zeros((num_layers,1000))  # [num_layers]

    inputs, targets = hessian_data
    for i in range(100):
        model(inputs[i:i+1])
        model.compute_hessian(None)
        for idx, value in enumerate(model.hessians()):
            # print(idx, value.max())
            select = value.detach().view(-1).sort(descending=True)[0][:10]
            hessians[idx,i*10:(i*10+10)] = select.cpu().numpy()

    return hessians, all_eigen

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    batch_size = 128
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

    train_h_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

    model = ResNet18_hessian(len(train_dataset.classes)).cuda()
    criterion = nn.CrossEntropyLoss()
    regularizer = corr_regularizer()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.1)

    for inputs, labels in train_h_loader:
        hessian_data = (inputs.cuda(), labels.cuda())
        break

    t = time.time()
    for epoch in range(50):
        print('----------------start train-----------------')
        train(model, train_loader, optimizer, regularizer, epoch)
        lr_scheduler.step()
        print('----------------end train-----------------')

        print('----------------start test-----------------')
        evaluate(model, test_loader)
        print('----------------end test-----------------')

        print('----------------start saving data-----------------')
        hessians, eigens = compute_hessain_value(model, criterion, hessian_data)
        # np.save("./checkpoint/cifar/FOR_E_"+str(epoch+1)+".npy", hessians)
        # np.save("./checkpoint/cifar/EST_E_"+str(epoch+1)+".npy", eigens)


    print('time:', time.time() - t)






