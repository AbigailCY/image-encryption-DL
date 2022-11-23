""" Utilities """
import os
import logging
import shutil
import torch
import numpy as np
import multiprocessing
import torch.nn.functional as F
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def get_logger(file_path, distributed_rank=0):
    """ Make python logger """
    logger = logging.getLogger('palm_cnn')
    if distributed_rank > 0:
        return logger
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def param_size(model):
    """ Compute parameter size in MB """
    n_params = sum(np.prod(v.size()) for k, v in model.named_parameters())
    return n_params / 1024. / 1024.


def save_checkpoint(state, ckpt_dir, is_best=False, epoch=0):
    filename = os.path.join(ckpt_dir, 'checkpoint_{}.pth.tar'.format(epoch))
    torch.save(state, filename)
    last_filename = os.path.join(ckpt_dir, 'last.pth.tar')
    shutil.copyfile(filename, last_filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def rankn(label, dist, n=1):
    idx = dist.argsort(dim=1)
    gt = label.view(label.shape[0], 1) == label.view(1, label.shape[0])
    count = 0
    for i in range(gt.shape[0]):
        if True in gt[i][idx[i][1:n+1]]:
            count += 1

    rank_n = count/gt.shape[0]
    return rank_n


def eer(label, dist):
    gt = label.view(label.shape[0], 1) == label.view(1, label.shape[0])
    score = 1 - dist
    eers = []
    for i in range(gt.shape[0]):
        fpr, tpr, thresholds = roc_curve(gt[i].cpu().numpy(), score[i].cpu().numpy())
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        # thresh = interp1d(fpr, thresholds)(eer)
        eers.append(eer)

    eer = sum(eers)/gt.shape[0]
    return eer


def process_map(r):
    return torch.Tensor([torch.mean(r[:i + 1].float()) for i in range(r.shape[0]) if r[i]])


def map(label, dist):
    if len(label) > 1000:
        idx = dist.argsort(dim=1)
        gt = label.view(label.shape[0], 1) == label.view(1, label.shape[0])
        rs = [gt[i][idx[i]] for i in range(gt.shape[0])]

        pool = multiprocessing.Pool(4)
        result = []
        for r in rs:
            p = pool.apply_async(process_map, args=(r.cpu(),))
            result.append(p)
        pool.close()
        pool.join()

        trec_precisions = []
        for r in result:
            trec_precisions.append(r.get())

        trec_precisions = torch.cat(trec_precisions)
        mAP = torch.mean(trec_precisions)

    else:
        idx = dist.argsort(dim=1)
        gt = label.view(label.shape[0], 1) == label.view(1, label.shape[0])
        rs = [gt[i][idx[i]] for i in range(gt.shape[0])]
        trec_precisions = []
        for r in rs:
            trec_precision = torch.Tensor([torch.mean(r[:i+1].float()) for i in range(r.shape[0]) if r[i]])
            trec_precisions.append(trec_precision)

        trec_precisions = torch.cat(trec_precisions)
        mAP = torch.mean(trec_precisions)

    return mAP.item()


def iou(box1, box2):
    in_h = torch.min(box1[:,2], box2[:,2]) - torch.max(box1[:,0], box2[:,0])
    in_w = torch.min(box1[:,3], box2[:,3]) - torch.max(box1[:,1], box2[:,1])
    inter = in_h*in_w
    inter[inter<=0] = 0
    union = (box1[:,2] - box1[:,0])*(box1[:,3] - box1[:,1]) + (box2[:,2] - box2[:,0])*(box2[:,3] - box2[:,1]) - inter
    iou = inter / union
    return iou.mean()


def nms_centernet(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep