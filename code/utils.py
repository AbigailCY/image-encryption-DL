# adjust
def adjust_learning_rate(initial_lr, optimizer, epoch, every_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // every_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# eval
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# misc
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import torch
# mean_r = torch.rand(3)
# mean_i = torch.rand(3)
# a = torch.stack((mean_r,mean_i),dim=1)
# print(a.shape)
import pandas
splits = pandas.read_csv(("./list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
torch.as_tensor(identity[mask].values)