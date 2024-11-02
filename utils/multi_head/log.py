#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import Dataset, Sampler

class log_dataset(Dataset):
    def __init__(self, logs, labels, seq=True, quan=False, sem=False):
        self.seq = seq
        self.quan = quan
        self.sem = sem
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
        if self.sem:
            self.Semantics = logs['Semantics']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        if self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                            dtype=torch.float)
        return log, self.labels[idx]


class experience_replay:
    def __init__(self):
        self.experience_pool = defaultdict(str)

    def to_str(self, obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.squeeze()
            obj = ' '.join(list(map(str, map(int,obj.tolist()))))
        elif isinstance(obj, list):
            obj = ' '.join(list(map(str, obj)))
        elif isinstance(obj, tuple):
            obj = ' '.join(list(map(str, list(obj))))
        return obj

    def add_sequence(self, sequence, candidate):
        sequence = self.to_str(sequence)
        candidate = self.to_str(candidate)
        self.experience_pool[sequence]= candidate
    def find_sequence(self, sequence):
        result = None
        sequence = self.to_str(sequence)
        is_exist = False
        if (sequence in self.experience_pool.keys()):
            is_exist =True
            candidate = self.experience_pool[sequence]
            result = list(map(int, candidate.strip().split()))
        return is_exist, result


if __name__ == '__main__':
    data_dir = '../../data/hdfs/hdfs_train'
    window_size = 10
    train_logs = prepare_log(data_dir=data_dir,
                             datatype='train',
                             window_size=window_size)
    train_dataset = log_dataset(log=train_logs, seq=True, quan=True)
    print(train_dataset[0])
    print(train_dataset[100])
