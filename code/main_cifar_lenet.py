import os
import torch
from torch import nn
from LeNet_cifar import LeNet
from torchvision import datasets, transforms
import time

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        model.compute_hessian()

        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = data.cuda(), target.cuda()
        output = model(data)

        test_loss += criterion(output, target)
        pred = output.max(1, keepdim=True)[1]                           #
        correct += pred.eq(target.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
       test_loss, correct, len(test_loader.dataset),
       100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    batch_size = 64
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )
    train_dataset = datasets.CIFAR10(root='./', train=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # model = LeNet().cuda()
    model = LeNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

    start = time.time()
    for epoch in range(1, 21):
        print('----------------start train-----------------')
        
        train(model, train_loader, optimizer, epoch)
        print('----------------end train-----------------')

        print('----------------start test-----------------')
        test(model, test_loader)
        print('----------------end test-----------------')
    print(time.time()-start)

