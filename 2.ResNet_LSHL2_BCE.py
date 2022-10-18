# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:2.ResNet_LSHL2_BCE.py
@Time:2022/10/6 20:40

"""
import time

import torch.nn as nn
import torch.optim as optim

from core import ramps
from core.config import *
from core.eval import validate, cal_f1
from core.utils import parameters_string, load_pretrained, create_model, save_checkpoint, AverageMeterSet, fix_BN_stat, \
    cal_for_batch
from data_prepare import *
from losses.L1 import L1
from losses.L2 import L2
from losses.LSH import LSH
from teacher_model.ConvMixer import ConvMixer_768_32


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    if args.lr_rampup != 0:
        lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    if args.lr_reduce_epochs:
        reduce_epochs = [int(x) for x in args.lr_reduce_epochs.split(',')]
        for ep in reduce_epochs:
            if epoch >= ep:
                lr /= 2

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, model_t, class_criterion, criterion_kd, optimizer, epoch):
    global global_step
    start_time = time.time()

    meters = AverageMeterSet()

    Sig = torch.nn.Sigmoid()
    # switch to train mode
    model.train()
    if args.fix_BN_stat:
        model.apply(fix_BN_stat)
    # teacher model select eval mode
    model_t.eval()

    end = time.time()
    for batch_idx, (inputs, labels, age_genders,_) in enumerate(train_loader):
        inputs, labels, age_genders = inputs.to(dtype=torch.float32), labels.to(dtype=torch.float32), \
                                      age_genders.to(dtype=torch.float32)
        # measure data loading time
        meters.update('data_time', time.time() - end)

        # adjust_learning_rate(optimizer, epoch, batch_idx, len(train_loader))


        inputs, labels,age_genders = inputs.cuda(), labels.cuda(),age_genders.cuda()

        feat_s, logit_s = model(inputs, age_genders, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(inputs, age_genders, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = class_criterion(logit_s, labels)

        f_s = feat_s[-1]
        f_t = feat_t[-1]
        if args.distill == 'lshl2_s':
            loss_kd = criterion_kd(f_s, f_t, logit_t, labels)
        else:
            loss_kd = criterion_kd(f_s, f_t)

        loss = args.gamma * loss_cls + args.beta * loss_kd

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        # cal_acc
        output = Sig(logit_s)
        acc, _ = cal_for_batch(output.data, labels, args.thre)
        meters.update('lr', optimizer.param_groups[0]['lr'])
        meters.update('loss', loss.item())
        meters.update('class_loss', loss_cls.item())
        meters.update('kd_loss', loss_kd.item())
        meters.update("acc", float(acc), inputs.size(0))
        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if batch_idx % args.print_freq == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Loss {meters[loss]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'KD {meters[kd_loss]:.4f}\t'
                "Acc {meters[acc]:.4f}\t".format(epoch, batch_idx, len(train_loader), meters=meters))
    logger.info("*TRAIN Acc {:.3f}  ({:.1f}/{:.1f})".format(meters['acc'].avg, meters['acc'].sum / 100, meters['acc'].count))
    # if writer is not None:
    #     writer.add_scalar("train/lr", meters['lr'].avg, epoch)
    #     writer.add_scalar("train/loss", meters['loss'].avg, epoch)

    logger.info("--- training epoch in {} seconds ---".format(time.time() - start_time))

    return meters


if __name__ == '__main__':
    global_step = 0
    # general
    args = get_args_parser()
    # change setting for KD
    args.beta = 0.5
    args.pretrained_s = "pretrained/feat_pretrained/checkpoint.final.ckpt"
    # set random_seed
    seed_everything(args.seed)
    # data prepared
    start = time.time()
    datas, label, ex_feat = CategoryDataset.make_data_loading(args.data_path)
    if args.wandb:
        import wandb

        anony = "must"


        def class2dict(f):
            return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


        run = wandb.init(project='KD_20class',
                         name="LHSL2_KD_" + datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
                         config=class2dict(args),
                         group='LHSL2_KD',
                         job_type="train",
                         anonymous=anony)
    #  debug enable
    if args.debug:
        datas = datas[:10]
        label = label[:10]
        ex_feat = ex_feat[:10]
        args.epochs = 2
    # save experiments config and log
    filename = os.path.basename(__file__).split(".")[1]
    set_save_path(args, filename)
    write_settings(args)
    # checkpoint path
    checkpoint_path = args.save_path + "/checkpoint"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    args.checkpoint_path = checkpoint_path
    # args.save_path
    logger = get_logger(args.save_path, "main")
    #  data split
    all_data = CategoryDataset.make_data_split(datas, label, ex_feat)
    #  train_data prepared
    train_dataset = CategoryDataset(all_data["train_data"][0], all_data["train_label"], all_data["train_data"][1])
    train_loader = CategoryDataset.create_train_loader(train_dataset, args=args)
    val_dataset = CategoryDataset(all_data["val_data"][0], all_data["val_label"], all_data["val_data"][1])
    val_loader = CategoryDataset.create_eval_loader(val_dataset, args=args)
    logger.info(f"=> Loading Finish,waste: {time.time() - start} seconds")
    #  teacher model prepared
    #  logger.info(f"=> creating teacher model '{args.arch_t}'")
    model_t = ConvMixer_768_32(20).cuda()
    logger.info(f"=> Create Teacher Model '{type(model_t).__name__}'")
    logger.info(parameters_string(model_t))
    model_t = load_pretrained(model_t, args.pretrained_t, None, logger,DataParallel=False)
    # TODO: shuffle_last_conv_classifier,how it work?
    model_t.eval()

    weight, bias = model_t.get_classifier_weight()
    t_dim = weight.shape[1]
    logger.info('=> teacher feature dim: {}'.format(t_dim))
    logger.info('=> teacher classifier weight std: {}'.format(weight.std()))
    if args.std is None:
        args.std = weight.std()
    if args.hash_num is None:
        args.hash_num = 4 * t_dim

    # create student model
    model_s = create_model(args.stu_arch, args.num_classes, DataParallel=False, student_dim=t_dim,
                           force_2FC=args.force_2FC, change_first=True)
    logger.info(f"=> Create Student Model '{args.stu_arch}'")
    logger.info(parameters_string(model_s))

    model_s = load_pretrained(model_s, args.pretrained_s, args.stu_arch, logger,DataParallel=False)

    logger.info('=> creating {} for knowledge distillation'.format(args.distill))
    if args.distill == 'l1':
        criterion_kd = L1()
    elif args.distill == 'l2':
        criterion_kd = L2()
    elif args.distill == 'lsh':
        logger.info('=> LSH: D:{} N:{} std:{} LSH_loss:{}'.format(t_dim, args.hash_num, args.std, args.class_criterion))
        criterion_kd = LSH(t_dim, args.hash_num, args.std, with_l2=False, LSH_loss=args.class_criterion)
    elif args.distill == 'lshl2':
        logger.info(
            '=> LSHl2: D:{} N:{} std:{} LSH_loss:{}'.format(t_dim, args.hash_num, args.std, args.class_criterion))
        criterion_kd = LSH(t_dim, args.hash_num, args.std, with_l2=True, LSH_loss=args.class_criterion)
    else:
        raise NotImplementedError(args.distill)
    # logger.info("=> creating student model '{}'".format(type(model_s).__name__)

    criterion_kd = criterion_kd.cuda()

    logger.info('=> creating {} for class criterion'.format(args.class_criterion))
    if args.class_criterion == 'BCE':
        class_criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(args.class_criterion)

    if args.finetune_fc:
        paras = model_s.feat_fc.parameters()
    else:
        paras = model_s.parameters()

    optimizer = optim.AdamW(paras, lr=args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        args.best_F1 = checkpoint["best_f1"]
        model_s.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    if 'lsh' in args.distill:
        if args.bias == '0':
            logger.info('=> init LSH bias by 0')
        elif args.bias == 'median':
            logger.info('=> init LSH bias by median')
            criterion_kd.init_bias(model_t, train_loader, args.print_freq, use_median=True)
        elif args.bias == 'mean':
            logger.info('=> init LSH bias by mean')
            criterion_kd.init_bias(model_t, train_loader, args.print_freq, use_median=False)
        else:
            raise NotImplementedError(args.bias)

    logger.info('=> evaluate teacher')
    args.cal_F1 = 0
    t_pred_val, t_label_val,_ = validate(val_loader, model_t, logger, args)
    acc, F1, F1_all = cal_f1(t_pred_val, t_label_val, 0.5)
    logger.info(f'=> Accuracy_val_gross:, {acc}, F1_val_gross:, {F1}, F1_val_all:, {F1_all}')
    # early stop

    # todo: tonight d0
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_meters=train(train_loader, model_s, model_t, class_criterion, criterion_kd, optimizer, epoch)
        pred_val, label_val,val_meters  = validate(val_loader, model_s, logger, args)
        acc, F1, F1_all= cal_f1(pred_val, label_val, 0.5)
        logger.info(f'=> Evaluate Student Accuracy_val_gross:, {acc}, F1_val_gross:, {F1}, F1_val_all:, {F1_all}')
        if args.wandb:
            wandb.log({f"KD Epoch": epoch + 1,
                       f"KD avg_train_loss": train_meters["class_loss"].avg,
                       f"KD avg_val_loss":  val_meters["loss"].avg,
                       })
        is_best = F1 > args.best_F1
        if is_best:
            args.best_F1 = F1
            torch.save(model_s.state_dict(), args.checkpoint_path + "/bst_F1.pt")
            logger.info('\t')
            logger.info("=> save bst F1 model")
        #  resume training
        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.stu_arch,
                'state_dict': model_s.state_dict(),
                'best_f1': args.best_F1,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1, logger)
    save_checkpoint({
        'epoch': epoch + 1,
        'global_step': global_step,
        'arch': args.stu_arch,
        'state_dict': model_s.state_dict(),
        'best_f1': args.best_F1,
    }, False, checkpoint_path, 'final', logger)
    logger.info("*"*30)
    logger.info("best_f1 {}".format(args.best_F1))