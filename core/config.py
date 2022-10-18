# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:config.py
@Time:2022/9/27 16:01

"""
import argparse
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args_parser():
    parser = argparse.ArgumentParser(description='category_KD')

    # general
    parser.add_argument("--save_path", type=str, default="", help="output directory")
    parser.add_argument("--seed", type=int, default=2022, help="seed")
    parser.add_argument("--checkpoint_path", type=str, default="", help="model save path")
    parser.add_argument("--device", type=str, default="", help="run_device")
    parser.add_argument("---num_class", type=int, default=2, help="class_num")  # data
    parser.add_argument("--wandb", type=int, default=0, help="use_wandb")
    parser.add_argument("--debug", type=bool, default=False, help="Debug")
    parser.add_argument("--resume", type=bool, default=False, help="resume")
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.today().strftime("%Y%m%d"),
        help="date of experiment",
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="suffix of save dir name"
    )
    # specific
    parser.add_argument("--data_path", type=str, default="data", help="path of dataset")
    parser.add_argument("--stu_arch", type=str, default='resnet18', help="path of dataset")
    parser.add_argument("--pretrained_s", type=str, default="", help='path of student model')
    parser.add_argument("--pretrained_t", type=str, default="pretrained/teacher_pretrained/2022-07-07 23-52-37.pt", help='path of teacher model')
    parser.add_argument("--distill", type=str, default="lshl2", help="distill loss")
    parser.add_argument("--apex", type=bool, default=True, help="AMP_ENABLE")
    parser.add_argument(
        "--workers",
        type=int,
        default="4",
        help="number of threads used for data loading",
    )
    parser.add_argument("--std", default=None, help="std of teacher dim")
    parser.add_argument("--gamma", default=1, help="loss hyperparameter gamma")
    parser.add_argument("--beta", default=1, help="loss  hyperparameter  beta")
    parser.add_argument("--print_freq", default=100, help="output frequent")
    parser.add_argument("--hash_num", default=None, help="hash_num of teacher")
    parser.add_argument("--bias", default="median", help="hash_bias")
    parser.add_argument("--num_classes", default=20, help="classification member")
    parser.add_argument("--force_2FC", default=True, help="the using way of student last layer ")
    parser.add_argument("--best_F1", default=0, help="Best F1")
    parser.add_argument("--cal_F1", default=0, help="calculate F1")
    # optimization
    parser.add_argument("--class_criterion", default='BCE', help="criterion")
    parser.add_argument("--epochs", type=int, default=50, help="run epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="mini-batch size")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum term")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--start_epoch",type=int,default=0,help="start epoch")
    parser.add_argument("--checkpoint_epochs", type=int, default=15, help="checkpoint_epochs")
    parser.add_argument("--finetune_fc",type=bool, default=False, help="finetune fc layer")
    parser.add_argument("--lr_mode",type=str,default="cosine",help="using lr_mode")
    parser.add_argument("--lr", type=float, default=5e-4
                        , help="initial learning rate")
    parser.add_argument("--lr_rampup", default=0
                        , help="lr_rampup")
    parser.add_argument("--lr_rampdown_epochs", default=100
                        , help="lr_rampdown_epochs")
    parser.add_argument("--lr_reduce_epochs",default=0,help="lr_reduce_epochs")
    parser.add_argument(
        "--lr_scheduler_type", type=str, default="cosine", help="multi_step"
    )
    parser.add_argument('--warmup_epochs', help='warmup_epochs', type=int, default=2)
    parser.add_argument('--warmup_lr', help='warmup_lr', type=float, default=1e-5)
    parser.add_argument('--target_lr', help='target_lr', type=float, default=1e-4)
    parser.add_argument("--thre",type=int,default=0.5,help="using thre")
    parser.add_argument('--max_grad_norm', type=float, default=1000, help="grad_clip")
    parser.add_argument('--fix_BN_stat',default=False,help="fix BN_STAT")

    args = parser.parse_args()

    return args


def get_logger(save_path, logger_name):
    """
    Initialize logger
    """

    logger = logging.getLogger(logger_name)  # 获取logger对象，如果不指定name则返回root对像
    logger.propagate = False
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console_formatter = logging.Formatter("%(message)s")

    # file log
    file_handler = logging.FileHandler(os.path.join(save_path, "experiment.log"))
    file_handler.setFormatter(file_formatter)

    # console log
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.setLevel(logging.INFO)
    return logger


def create_dir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)


def set_save_path(args, filename):

    if filename.endswith("train_feat_fc"):
        dirs = "train_feat_fc"
    elif filename.endswith("BCE"):
        dirs = "BCE_KD"
    else:
        dirs = "Base_Stu"
    if len(args.suffix) == 0:
        suffix = "log_{}_bs{:d}_epoch{:d}_lr{:.5f}/".format(
            args.date,
            args.batch_size,
            args.epochs,
            args.lr,
        )
    else:
        suffix = args.suffix
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        args.save_path = os.path.join("log", dirs, suffix)
    else:
        args.save_path = os.path.join("log", dirs, suffix)
    create_dir(args.save_path)


def write_settings(settings):
    """
    Save expriment settings to a file
    :param settings: the instance of option
    """

    with open(os.path.join(settings.save_path, "settings.log"), "w") as f:
        for k, v in settings.__dict__.items():
            f.write(str(k) + ": " + str(v) + "\n")


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn
