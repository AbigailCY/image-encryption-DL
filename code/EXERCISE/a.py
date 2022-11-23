# #1
# import torch
# from torch.autograd import Variable # torch 中 Variable 模块
# import torch.nn.functional as F     # 激励函数都在这

# # 先生鸡蛋
# tensor = torch.FloatTensor([[1,2],[3,4]])
# # 把鸡蛋放到篮子里, requires_grad是参不参与误差反向传播, 要不要计算梯度
# variable = Variable(tensor, requires_grad=False)

# t_out = torch.mean(tensor*tensor)       # x^2
# v_out = torch.mean(variable*variable)   # x^2



# # 做一些假数据来观看图像
# x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
# x = Variable(x)

# x_np = x.data.numpy()


# y_relu = torch.relu(x).data.numpy()
# y_sigmoid = torch.sigmoid(x).data.numpy()
# y_tanh = torch.tanh(x).data.numpy()
# y_softplus = F.softplus(x).data.numpy()


# #2
# import torch
# import matplotlib.pyplot as plt
# import torch.nn.functional as F     # 激励函数都在这

# # x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# # y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)
# # # 画图
# # plt.scatter(x.data.numpy(), y.data.numpy())
# # plt.show()

# class Net(torch.nn.Module):  # 继承 torch 的 Module
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()     # 继承 __init__ 功能
#         # 定义每层用什么样的形式
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
#         self.predict = torch.nn.Linear(n_hidden, n_output)   # 输出层线性输出

#     def forward(self, x):   # 这同时也是 Module 中的 forward 功能
#         # 正向传播输入值, 神经网络分析出输出值
#         x = torch.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
#         x = self.predict(x)             # 输出值
#         return x

# net = Net(n_feature=2, n_hidden=10, n_output=2)
# # print(net)  # net 的结构
# # """
# # Net (
# #   (hidden): Linear (1 -> 10)
# #   (predict): Linear (10 -> 1)
# # )
# # """

# # optimizer 是训练的工具
# optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
# loss_func = torch.nn.CrossEntropyLoss()     # 预测值和真实值的误差计算公式 (均方差)





# # 假数据
# n_data = torch.ones(100, 2)         # 数据的基本形态
# x0 = torch.normal(2*n_data, 1)      # 类型0 x data (tensor), shape=(100, 2)
# y0 = torch.zeros(100)               # 类型0 y data (tensor), shape=(100, )
# x1 = torch.normal(-2*n_data, 1)     # 类型1 x data (tensor), shape=(100, 1)
# y1 = torch.ones(100)                # 类型1 y data (tensor), shape=(100, )

# # 注意 x, y 数据的数据形式是一定要像下面一样 (torch.cat 是在合并数据)
# x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
# y = torch.cat((y0, y1), ).type(torch.LongTensor)    # LongTensor = 64-bit integer


# # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
# # plt.show()
# plt.ion()   # 画图
# plt.show()

# # for t in range(200):

# #     prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值
# #     loss = loss_func(prediction, y)     # 计算两者的误差
# #     optimizer.zero_grad()   # 清空上一步的残余更新参数值
# #     loss.backward()
# #     optimizer.step()

# #     # 接着上面来
# #     if t % 5 == 0:
# #         # plot and show learning process
# #         plt.cla()
# #         plt.scatter(x.data.numpy(), y.data.numpy())
# #         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
# #         plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
# #         plt.pause(0.1)
# for t in range(200):

#     out = net(x)     # 喂给 net 训练数据 x, 输出预测值
#     loss = loss_func(out, y)     # 计算两者的误差
#     optimizer.zero_grad()   # 清空上一步的残余更新参数值
#     loss.backward()
#     optimizer.step()

#     # 接着上面来
#     if t % 5 == 0:
#         # plot and show learning process
#         plt.cla()
#         prediction = torch.max(F.softmax(out), 1)[1]
#         pred_y = prediction.data.numpy().squeeze()
#         target_y = y.data.numpy()

#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
#         accuracy = sum(pred_y == target_y)/200
#         plt.text(1.5, -7, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color':  'red'})
#         plt.pause(0.1)

# #3
# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(n_feature, n_hidden)
#         self.predict = torch.nn.Linear(n_hidden, n_output)

#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x

# net1 = Net(1, 10, 1)

# net2 = torch.nn.Sequencial(torch.nn.Linear(1,10),torch.nn.relu(),torch.nn.Linear(10,1))


# x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
# y = x.pow(2) + 0.2*torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)

# #4 
# def save():
#     # 建网络
#     net1 = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1)
#     )
#     optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
#     loss_func = torch.nn.MSELoss()

#     # 训练
#     for t in range(100):
#         prediction = net1(x)
#         loss = loss_func(prediction, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     torch.save(net1, 'net.pkl')  # 保存整个网络
#     torch.save(net1.state_dict(), 'net_params.pkl')   # 只保存网络中的参数 (速度快, 占内存少)

# def restore_net():
#     # restore entire net1 to net2
#     net2 = torch.load('net.pkl')
#     prediction = net2(x)

# def restore_params():
#     # 新建 net3
#     net3 = torch.nn.Sequential(
#         torch.nn.Linear(1, 10),
#         torch.nn.ReLU(),
#         torch.nn.Linear(10, 1)
#     )

#     # 将保存的参数复制到 net3
#     net3.load_state_dict(torch.load('net_params.pkl'))
#     prediction = net3(x)


# save()

# # 提取整个网络
# restore_net()

# # 提取网络参数, 复制到新网络
# restore_params()



#5
# import torch
# import torch.utils.data as Data
# torch.manual_seed(1)    # reproducible

# BATCH_SIZE = 5      # 批训练的数据个数

# x = torch.linspace(1, 10, 10)       # x data (torch tensor)
# y = torch.linspace(10, 1, 10)       # y data (torch tensor)

# # 先转换成 torch 能识别的 Dataset
# torch_dataset = Data.TensorDataset(x, y)

# # 把 dataset 放入 DataLoader
# loader = Data.DataLoader(
#     dataset=torch_dataset,      # torch TensorDataset format
#     batch_size=BATCH_SIZE,      # mini batch size
#     shuffle=True,               # 要不要打乱数据 (打乱比较好)
#     num_workers=2,              # 多线程来读数据
# )

# for epoch in range(3):   # 训练所有!整套!数据 3 次
#     for step, (batch_x, batch_y) in enumerate(loader):  # 每一步 loader 释放一小批数据用来学习
#         # 假设这里就是你训练的地方...

#         # 打出来一些数据
#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
#               batch_x.numpy(), '| batch y: ', batch_y.numpy())

# """
# Epoch:  0 | Step:  0 | batch x:  [ 6.  7.  2.  3.  1.] | batch y:  [  5.   4.   9.   8.  10.]
# Epoch:  0 | Step:  1 | batch x:  [  9.  10.   4.   8.   5.] | batch y:  [ 2.  1.  7.  3.  6.]
# Epoch:  1 | Step:  0 | batch x:  [  3.   4.   2.   9.  10.] | batch y:  [ 8.  7.  9.  2.  1.]
# Epoch:  1 | Step:  1 | batch x:  [ 1.  7.  8.  5.  6.] | batch y:  [ 10.   4.   3.   6.   5.]
# Epoch:  2 | Step:  0 | batch x:  [ 3.  9.  2.  6.  7.] | batch y:  [ 8.  2.  9.  5.  4.]
# Epoch:  2 | Step:  1 | batch x:  [ 10.   4.   8.   1.   5.] | batch y:  [  1.   7.   3.  10.   6.]
# """



# #6
# import torch
# import torch.utils.data as Data
# import torch.nn.functional as F

# torch.manual_seed(1)    # reproducible

# LR = 0.01
# BATCH_SIZE = 25
# EPOCH = 12

# # fake dataset
# x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# # plot dataset
# # plt.scatter(x.numpy(), y.numpy())
# # plt.show()

# # 使用上节内容提到的 data loader
# torch_dataset = Data.TensorDataset(x, y)
# loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,)

# # 默认的 network 形式
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(1, 20)   # hidden layer
#         self.predict = torch.nn.Linear(20, 1)   # output layer

#     def forward(self, x):
#         x = F.relu(self.hidden(x))      # activation function for hidden layer
#         x = self.predict(x)             # linear output
#         return x

# # 为每个优化器创建一个 net
# net_SGD         = Net()
# net_Momentum    = Net()
# net_RMSprop     = Net()
# net_Adam        = Net()
# nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# # different optimizers
# opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
# opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
# opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
# opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
# optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

# loss_func = torch.nn.MSELoss()
# losses_his = [[], [], [], []]   # 记录 training 时不同神经网络的 loss

# for epoch in range(EPOCH):
#     print('Epoch: ', epoch)
#     for step, (b_x, b_y) in enumerate(loader):
#         print("loader, ", step)

#         # 对每个优化器, 优化属于他的神经网络
#         for net, opt, l_his in zip(nets, optimizers, losses_his):
#             output = net(b_x)              # get output for every net
#             loss = loss_func(output, b_y)  # compute loss for every net
#             opt.zero_grad()                # clear gradients for next train
#             loss.backward()                # backpropagation, compute gradients
#             opt.step()                     # apply gradients
#             l_his.append(loss.data.numpy())     # loss recoder


# #7 CNN
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision      # 数据库模块
import matplotlib.pyplot as plt

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = True  # 如果你已经下载好了mnist数据就写上 False


# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
                                                    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,      # input height
                out_channels=16,    # n_filters
                kernel_size=5,      # filter size
                stride=1,           # filter movement/step
                padding=2,      # 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1
            ),      # output shape (16, 28, 28)
            nn.ReLU(),    # activation
            nn.MaxPool2d(kernel_size=2),    # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # 分配 batch data, normalize x when iterate train_loader
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients


test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')

"""
...
Epoch:  0 | train loss: 0.0306 | test accuracy: 0.97
Epoch:  0 | train loss: 0.0147 | test accuracy: 0.98
Epoch:  0 | train loss: 0.0427 | test accuracy: 0.98
Epoch:  0 | train loss: 0.0078 | test accuracy: 0.98
"""