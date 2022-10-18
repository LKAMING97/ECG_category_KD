# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:eval.py
@Time:2022/9/30 10:18

"""
import time

import numpy as np
import torch
import torch.nn as nn

from core.utils import AverageMeterSet, cal_for_batch


def validate(eval_loader, model, LOG, args):
    start_time = time.time()
    # # 验证指标初始化

    tt_y_true = list()
    tt_y_pred = list()

    meters = AverageMeterSet()
    model.eval()
    Sig = torch.nn.Sigmoid()
    Loss = nn.BCEWithLogitsLoss()
    end = time.time()
    preds = []
    targets = []
    for batch_idx, (inputs, labels, age_genders, _) in enumerate(eval_loader):
        inputs, labels, age_genders = inputs.to(dtype=torch.float32), labels.to(dtype=torch.float32), \
                                      age_genders.to(dtype=torch.float32)

        meters.update('data_time', time.time() - end)
        # compute output
        with torch.no_grad():
            output = model(inputs.cuda(), age_genders.cuda()).cpu()

        # for mAP calculation
        tt_y_true.append(labels.cpu().detach().numpy())
        # tt_y_pred.append(Sig(output).cpu().detach().numpy())
        tt_y_pred.append(output.cpu().detach().numpy())
        preds.append(output.cpu())
        targets.append(labels.cpu())
        output_loss = Loss(output, labels)
        acc, _ = cal_for_batch(output.data, labels, args.thre)
        minibatch_size = len(labels)
        waste = time.time() - end
        meters.update('acc', acc, minibatch_size)
        meters.update('loss', output_loss.item(), minibatch_size)

        meters.update('batch_time', waste)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            LOG.info(
                'Test: [{0}/{1}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Loss {meters[loss]:.4f}\t'
                'Acc  {meters[acc]:.3f}'
                    .format(batch_idx, len(eval_loader), meters=meters))
    tt_y_true = np.vstack(tt_y_true)
    tt_y_pred = np.vstack(tt_y_pred)

    # if args.cal_F1:
    #     pass
        # acc, F1, F1_all = cal_f1(tt_y_pred, tt_y_true, 0.5)
        # LOG.info('=> Accuracy_val_gross:', acc, 'F1_val_gross:', F1, 'F1_val_all:', F1_all)
        #
        # if F1 > args.best_F1:
        #     args.best_F1 = F1
        #     torch.save(model.state_dict(), args.checkpoint_path + "/bst_F1.pt")
        #     LOG.info('\t')
        #     LOG.info("=> save bst F1 model")

    LOG.info("--- testing epoch in {} seconds ---".format(time.time() - start_time))
    return tt_y_pred, tt_y_true,meters


def prec_recall_for_batch(output, target, thre):
    pred = output.gt(thre).long()
    this_tp = (pred + target).eq(2).sum()
    this_fp = (pred - target).eq(1).sum()
    this_fn = (pred - target).eq(-1).sum()
    this_tn = (pred + target).eq(0).sum()

    this_prec = this_tp.float() / (
            this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
    this_rec = this_tp.float() / (
            this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0
    return this_prec, this_rec


def cal_f1(y_pred, y_true, threshold):
    y_true = y_true.astype(int)
    y_pred = y_pred
    TP = np.sum(np.logical_and(np.equal(y_true, 1), (y_pred >= threshold)))
    FP = np.sum(np.logical_and(np.equal(y_true, 0), (y_pred >= threshold)))
    FN = np.sum(np.logical_and(np.equal(y_true, 1), (y_pred < threshold)))
    TN = np.sum(np.logical_and(np.equal(y_true, 0), (y_pred < threshold)))

    A = (TP + TN) / (TP + FP + FN + TN+0.01)
    P = TP / (TP + FP+0.01)
    R = TP / (TP + FN+0.01)
    F1 = 2 * P * R / (P + R+0.01)

    TP_1 = np.sum(np.logical_and(np.equal(y_true, 1), (y_pred >= threshold)), axis=0)
    FP_1 = np.sum(np.logical_and(np.equal(y_true, 0), (y_pred >= threshold)), axis=0)
    FN_1 = np.sum(np.logical_and(np.equal(y_true, 1), (y_pred < threshold)), axis=0)
    TN_1 = np.sum(np.logical_and(np.equal(y_true, 0), (y_pred < threshold)), axis=0)

    P_1 = TP_1 / (TP_1 + FP_1+0.01)
    R_1 = TP_1 / (TP_1 + FN_1+0.01)
    F1_1 = 2 * P_1 * R_1 / (P_1 + R_1+0.01)

    return A, F1, F1_1


def cal_acc(y_pred, y_true, threshold):
    y_true = y_true.astype(int)
    y_pred = y_pred
    TP = np.sum(np.logical_and(np.equal(y_true, 1), (y_pred >= threshold)))
    FP = np.sum(np.logical_and(np.equal(y_true, 0), (y_pred >= threshold)))
    FN = np.sum(np.logical_and(np.equal(y_true, 1), (y_pred < threshold)))
    TN = np.sum(np.logical_and(np.equal(y_true, 0), (y_pred < threshold)))

    A = (TP + TN) / (TP + FP + FN + TN)
    return A
