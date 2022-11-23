import os
import torch
from torch import nn

from torchvision import datasets, transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        ## -------------
        if (batch_idx == len(train_loader)-1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            xs = optimizer.param_groups[0]['params']
            ys = loss  # put your own loss into ys
            grads = torch.autograd.grad(ys, xs, create_graph=True)  # first order gradient
            grads2 = get_second_order_grad(grads, xs)  # second order gradient

            eigens = []
            for i in [0,1,4]:
                grad = grads2[i].detach()
                eigenvalue, _  = torch.eig(grad)
            #     # eigenvalue, _ = np.linalg.eig(grad)
                eigens.append(eigenvalue[:,0])
            #     print(i, torch.max(eigenvalue[:,0]), torch.min(eigenvalue[:,0]))

            # print('grads', len(grads))
            # for count, g in enumerate(grads):
            #     print(count, g.shape)
            # print('grads2', len(grads2))
            # 150 2400 30720 10080 840
            # eigenvalues = []
            # for count, g in enumerate(grads2):
            #     # print(count, g.shape)
            #     eigenvalue, _ = torch.eig(grads2[count])
            #     eigenvalue = eigenvalue.cpu().detach().numpy().reshape(-1)
            #     eigenvalues.append(eigenvalue)
            #     print(count, np.max(eigenvalue), np.min(eigenvalue))
            # eigenvalues = torch.eig(grads2[3])
            # print(eigenvalues)
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
    return eigens



def get_second_order_grad(grads, xs):
    grads2 = []
    for j, (grad, x) in enumerate(zip(grads, xs)):
        # start = time.time()
        print('2nd order on ', j, 'th layer')
        grad = torch.reshape(grad, [-1])
        grads2_tmp = torch.zeros(len(grad), len(grad)).cuda()
        for i, g in enumerate(grad):
            g2 = torch.autograd.grad(g, x, retain_graph=True)[0]
            g2 = torch.reshape(g2, [-1])
            grads2_tmp[i, :] = g2
        grads2.append(grads2_tmp)
        # print('Time used is ', time.time() - start)

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

    from LeNet8 import LeNet
    model = LeNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))

    toSave3 = np.zeros((4,840))
    toSave2 = np.zeros((4,2400))
    toSave1 = np.zeros((4,150))
    for epoch in range(1, 5):
        print('----------------start train-----------------')
        eigns = train(model, train_loader, optimizer, epoch)
        print('----------------end train-----------------')

        print('----------------start test-----------------')
        test(model, test_loader)
        print('----------------end test-----------------')
        toSave1[epoch-1,:] = eigns[0].cpu().numpy()
        toSave2[epoch-1,:] = eigns[1].cpu().numpy()
        toSave3[epoch-1,:] = eigns[2].cpu().numpy()
    np.save("./LM_conv1.npy", toSave1)
    np.save("./LM_conv2.npy", toSave2)
    np.save("./LM_fc3.npy", toSave3)

    # t = time.time()
    # layers = [' Layer FC3', ' Layer FC2', ' Layer FC1', ' Layer conv2', ' Layer conv1']
    # with PdfPages('./graph/'+'lenet_minst'+'_compute.pdf') as pdf: 
        # for epoch in range(1, 5):
        #     print('----------------start train-----------------')
        #     eigns = train(model, train_loader, optimizer, epoch)
        #     print('----------------end train-----------------')

        #     print('----------------start test-----------------')
        #     test(model, test_loader)
        #     print('----------------end test-----------------')

    #         plt.figure(figsize=(15, 4))
    #         plt.subplot(131)
    #         eigenvalue = eigns[0].cpu().numpy()[:-3]

    #         weights = np.ones_like(eigenvalue)/float(len(eigenvalue))
    #         plt.hist(eigenvalue, bins = 500, weights=weights)
    #         plt.xticks(fontsize=6)
    #         plt.yticks(fontsize=6)
    #         plt.subplot(131).set_title('Eopch: '+ str(epoch)+' Layer conv1')

    #         plt.subplot(132)
    #         eigenvalue = eigns[1].cpu().numpy()[:-3]

    #         weights = np.ones_like(eigenvalue)/float(len(eigenvalue))
    #         plt.hist(eigenvalue, bins = 500, weights=weights)
    #         plt.xticks(fontsize=6)
    #         plt.yticks(fontsize=6)
    #         plt.subplot(132).set_title('Eopch: '+ str(epoch)+' Layer conv2')

    #         plt.subplot(133)
    #         eigenvalue = eigns[2].cpu().numpy()[:-3]

    #         weights = np.ones_like(eigenvalue)/float(len(eigenvalue))
    #         plt.hist(eigenvalue, bins = 500, weights=weights)
    #         plt.xticks(fontsize=6)
    #         plt.yticks(fontsize=6)
    #         plt.subplot(133).set_title('Eopch: '+ str(epoch)+' Layer FC3')



    #         for i in range(5):
    #             grad = grads2[i].cpu().numpy()
    #             eigenvalue, _ = np.linalg.eig(grad)
    #             print(np.max(eigenvalue), np.min(eigenvalue))
    #             plt.subplot(151+i)
    #             weights = np.ones_like(eigenvalue)/float(len(eigenvalue))
    #             plt.hist(eigenvalue, bins = 100, weights=weights)
    #             plt.xticks(fontsize=6)
    #             plt.yticks(fontsize=6)
    #             plt.subplot(151+i).set_title('Eopch: '+ str(epoch)+layers[i])
            # plt.tight_layout()
            # pdf.savefig()
            # plt.close()



            



    # print('time:', time.time()-t)
