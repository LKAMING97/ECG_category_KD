# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author:lkaming
@File:data_prepare.py
@Time:2022/9/29 11:28

"""
import os
from itertools import chain

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection._split import _validate_shuffle_split, train_test_split
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from torch.utils.data.dataset import Dataset


def multilabel_train_test_split(*arrays,
                                test_size=None,
                                train_size=None,
                                random_state=None,
                                shuffle=True,
                                stratify=None):
    """
    Train test split for multilabel classification. Uses the algorithm from:
    'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.
    """
    if stratify is None:
        return train_test_split(*arrays, test_size=test_size, train_size=train_size,
                                random_state=random_state, stratify=None, shuffle=shuffle)

    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"

    n_arrays = len(arrays)
    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    cv = MultilabelStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=123)
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )


class CategoryDataset(Dataset):
    def __init__(self, datas, labels, age_genders):
        self.datas = datas
        self.labels = labels
        self.age_genders = age_genders
        self.cas_dic = dict()
        self.class_num = labels.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx], self.age_genders[idx], idx

    @staticmethod
    def make_data_loading(data_path):
        global _datas, _label, _ex_feat

        for f in os.listdir(data_path):
            if f.startswith("data"):
                _datas = np.load(os.path.join(data_path, f))
            elif f.startswith("label"):
                _label = np.load(os.path.join(data_path, f))
            elif f.startswith("age_gender"):
                _ex_feat = np.load(os.path.join(data_path, f))

        return _datas, _label, _ex_feat

    @staticmethod
    def make_data_split(_data, _label, _feat):
        # split test
        tr_d, tt_idx, tr_l, _ = multilabel_train_test_split(np.arange(len(_data)).reshape(-1, 1), _label, stratify=_label, test_size=0.1)
        # split train valid
        tr_idx, val_idx, _, _ = multilabel_train_test_split(tr_d, tr_l, stratify=tr_l, test_size=0.2)
        tr_idx, val_idx, tt_idx = list(tr_idx.flatten()), list(val_idx.flatten()), list(tt_idx.flatten())

        train_data, val_data, test_data = [_data[tr_idx, :], _feat[tr_idx, :]], [_data[val_idx, :], _feat[val_idx, :]], \
                                          [_data[tt_idx, :], _feat[tt_idx, :]]
        train_label, val_label, test_label = _label[tr_idx, :], _label[val_idx, :], _label[tt_idx, :]

        _all_data = dict(train_data=train_data, train_label=train_label, val_data=val_data, val_label=val_label,
                         test_data=test_data, test_label=test_label)
        np.save("all_label.npy",dict(train_label=train_label, val_label=val_label,test_label=test_label))
        return _all_data

    @staticmethod
    def create_train_loader(train_dataset, args):
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
        return train_loader

    @staticmethod
    def create_eval_loader(eval_dataset, args):
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False)
        return eval_loader
