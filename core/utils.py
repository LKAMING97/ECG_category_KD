# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:utils.py
@Time:2022/9/29 16:14

"""

import os
import shutil

import numpy as np
import torch
import torch.nn as nn

from core.config import export
from student_model.ResNet_torch import ResNet, BasicBlock


@export
def resnet18(num_classes=20, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)
    return model


def create_model(model_name, num_classes, detach_para=False, DataParallel=True, **kwargs):
    model_factory = globals()[model_name]
    model_params = dict(num_classes=num_classes, **kwargs)
    model = model_factory(**model_params)
    if DataParallel:
        model = nn.DataParallel(model.cuda())
    else:
        model = model.cuda()

    if detach_para:
        for param in model.parameters():
            param.detach_()
    return model


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def load_pretrained(model, pretrained, arch, LOG, DataParallel=True):
    if os.path.isfile(pretrained):
        LOG.info("=> loading pretrained from checkpoint {}".format(pretrained))
        if DataParallel:
            model = nn.DataParallel(model)
        checkpoint = torch.load(pretrained)
        if pretrained.split("/")[1].startswith("teacher"):
            state_dict = checkpoint
        else:
            state_dict = checkpoint['state_dict']

        sd_arch = []
        if 'moco' in sd_arch:
            if DataParallel:
                replace_str = 'encoder_q.'
            else:
                replace_str = 'module.encoder_q.'
            state_dict.pop('module.encoder_q.fc.0.weight')
            state_dict.pop('module.encoder_q.fc.0.bias')
            state_dict.pop('module.encoder_q.fc.2.weight')
            state_dict.pop('module.encoder_q.fc.2.bias')
            state_dict = {k.replace(replace_str, ''): v for k, v in state_dict.items()}
            ret = model.load_state_dict(state_dict, strict=False)
        elif 'simclr' in sd_arch:
            if DataParallel:
                state_dict = {'module.{}'.format(k): v for k, v in state_dict.items()}
            ret = model.load_state_dict(state_dict, strict=False)
        else:
            ret = model.load_state_dict(state_dict, strict=False)
        LOG.info("=> loaded pretrained {}".format(pretrained))
        LOG.info("=> not load keys {} ".format(ret))
    elif pretrained.startswith('http'):
        state_dict = torch.hub.load_state_dict_from_url(pretrained, map_location='cpu')
        if 'vgg' in arch:
            state_dict.pop('classifier.6.weight')
            state_dict.pop('classifier.6.bias')
        elif 'mobilenet' in arch:
            state_dict.pop('classifier.1.weight')
            state_dict.pop('classifier.1.bias')
        elif 'densenet' in arch:
            state_dict.pop('classifier.weight')
            state_dict.pop('classifier.bias')
        elif 'fish' in arch:
            state_dict = state_dict['state_dict']
            state_dict.pop('module.fish.fish.9.4.1.weight', None)
            state_dict.pop('module.fish.fish.9.4.1.bias', None)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif 'se_res' in arch:
            state_dict.pop('last_linear.weight')
            state_dict.pop('last_linear.bias')
        elif 'efficientnet' in arch:
            state_dict.pop('_fc.weight')
            state_dict.pop('_fc.bias')
        elif 'hrnet' in arch:
            state_dict.pop('classifier.weight')
            state_dict.pop('classifier.bias')
        elif 'ViT' in arch:
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
        else:
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        if DataParallel:
            model = nn.DataParallel(model)
        LOG.info("=> loaded pretrained {} ".format(pretrained))
        LOG.info("=> not load keys {} ".format(ret))
    else:
        if DataParallel:
            model = nn.DataParallel(model)
        LOG.info("=> NOT load pretrained")
    return model


class MultiCrossEntropy(nn.Module):
    def __init__(self):
        super(MultiCrossEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        logit = self.logsoftmax(input)
        loss = - logit * target
        loss = loss.sum() / target.sum()
        return loss


def save_checkpoint(state, is_best, dirpath, epoch, LOG) :
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def fix_BN_stat(module):
    classname = module.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        module.eval()


def fix_BN_learn(module):
    classname = module.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        module.weight.requires_grad_(False)
        module.bias.requires_grad_(False)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1.):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1.):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def cal_for_batch(output, target, thre):
    pred = output.gt(thre).long()
    this_tp = (pred + target).eq(2).sum() # 预测为正，实则为正
    this_fp = (pred - target).eq(1).sum() # 预测为正，实则为负
    this_fn = (pred - target).eq(-1).sum() # 预测为负，实则为正
    this_tn = (pred + target).eq(0).sum()
    # 计算精确率
    this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
    # 计算召回率
    this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0
    # 计算准确率
    this_acc = (this_tp.float() + this_tn.float()) / (
                this_tp + this_fp + this_fn + this_tn).float() * 100.0 if this_tp + this_fp + this_fn + this_tn != 0 else 0.0
    return this_acc, this_rec


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, LOG):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            LOG.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model,LOG)
            self.counter = 0

    def save_checkpoint(self, val_loss, model,LOG):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            LOG.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), 'checkpoint.pt')	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss





if __name__ == '__main__':
    model_s = create_model('resnet18', 20, DataParallel=False, student_dim=800,
                           force_2FC=True, change_first=True)
    a = torch.randn((1, 12, 5000)).cuda()
    b = torch.randn((1, 5)).cuda()
    c = model_s(a, b, is_feat=True)
