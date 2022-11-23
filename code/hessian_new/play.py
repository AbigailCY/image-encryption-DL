
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



def gmean(x):
    n = x.shape[0]
    x1 = 1/x
    return n/x1.sum(0)

a = torch.randn(3,4,5)
print(a)
b = a.mean(0)
print(b.shape)
print(b)
print(a.shape)
b = gmean(a)
print(b.shape)
print(b)

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])
    # return [torch.sum(x * y) for (x, y) in zip(xs, ys)]

def group_product1(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return [torch.sum(x * y) for (x, y) in zip(xs, ys)]

def group_add1(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha[i])
    return params

def group_add2(params, update, alpha):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(-update[i] * alpha[i])
    return params

params = [torch.rand(10,20), torch.rand(20, 40)]
v = [torch.ones_like(p) for p in params]
# print(params[0][2,2])
alpha = [torch.ones(1)*3, torch.ones(1)*4]

# print(group_add1(params,v,alpha)[0][2,2])
# alpha = [-torch.ones(1)*3, -torch.ones(1)*4]
# print(group_add2(params,v,alpha)[0][2,2])

# print(group_product1(params,v))

# alpha = group_product1(params,v)

# beta = [torch.sqrt(alpha[i])for i in range(len(alpha))]
# print(beta)

# a = np.random.rand(10,15,100)
# print(a[:,1,:].shape)

def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v

# a = [torch.tensor([1,2]),torch.tensor([3,4,5])]
# def o(abc):
#     # for i, p in enumerate(abc):
#     #     abc[i] = normalization(abc[i])
#     return [normalization([abc[i]])[0] for i in range(len(abc))]
    
# c = [normalization([a[i]])[0] for i in range(len(a))]
# print(c)
# print(a)
# c = o(a)
# print(c)
# print(normalization(a))
# b = a
# for i, p in enumerate(a):
#     b[i] = normalization(a[i])
# print(a)
# print(b)

# c = (normalization(a[i]) for i in range(len(a)))




# a = [1,2,3]
# b = [2,3,4]

# for i in range(len(a)):
#     print(a[i],b[i])




# fig=plt.figure('subplot demo')
# # row 3 col 2
# with PdfPages('./play.pdf') as pdf: 
#     for epoch in range(5):
#         fig = plt.figure(figsize=(5*18,4))
#         for i in range(1,19):
#             fig.add_subplot(1, 18, i).set_title('Eopch: '+ str(epoch)+' Layer '+str(i))
#             # plt.subplot(131).set_title('Eopch: '+ str(epoch)+' Layer FC1')
#             x = np.arange(0, 3 * np.pi, 0.1)    # 创建一个一维数组，[0, 3*pi),步长为0.1
#             y = np.sin(x)
#             plt.ylabel('Density (Log Scale)', fontsize=8, labelpad=8)
#             plt.xlabel('Eigenvlaue', fontsize=8, labelpad=8)
#             plt.plot(x,y)
#         plt.tight_layout()
#         pdf.savefig()
#         plt.close()




# grids = np.linspace(0,2,1000)
# density = np.sin(grids)

# fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(2, 2, 1)
# plt.plot(grids, density)
# ax.set_xscale('log')
# ax.set_yscale('log')
# fig.add_subplot(2, 2, 2)
# plt.semilogx(grids, density + 1.0e-7)
# fig.add_subplot(2, 2, 3)
# plt.semilogx(grids+ 1.0e-7, density)
# fig.add_subplot(2, 2, 4)
# plt.loglog(grids, density)
# # plt.xlim(left=1e-1)


# plt.tight_layout()
# plt.show()


# a = np.array([1,2,3,4,5,6,5,5,5,7])
# a = np.delete(a, np.argwhere(a<2))
# print(a)


# a = np.zeros([1,100])
# b = list(a)
# c = b[0]
# print(len(c))