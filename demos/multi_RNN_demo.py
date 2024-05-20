#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.append('../')
import torch
from models.multiHeadRNN import multiHead as MidLog
from tools.multi_head_tools import predicter as Predicter
from tools.multi_head_tools import trainer as Trainer
from utils.multi_head.utils import seed_everything
# Config Parameters
options = dict()
options['data_dir'] = '../data/multiHeadRNN/'
options['window_size'] = 10
options['device'] = 'cuda'if torch.cuda.is_available()else 'cpu'

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# Features
options['sequentials'] = True
options['quantitatives'] = False
options['semantics'] = False
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 1
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 28

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1
options['train_data'] = 'hdfs_train','hdfs_test_normal' # The latter serves as validation
options['dataset_name'] = 'HDFS'
options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 370
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "midlog"
options['save_dir'] = "../results/multi_head/"
options['bidirectional'] = False
options['test_data'] = 'hdfs_test_normal','hdfs_test_abnormal'
# Predict
options['model_path'] = "../results/multi_head/midlog_bestloss.pth"
options['num_candidates'] = 10
# Config Parameters
seed_everything(seed=1234)

def train():
    Model =MidLog (model_name='gru'
                   , n_model= 4
                   , d_embd= 64
                   , n_events=options['num_classes']
                   , hidden_size = options['hidden_size']
                   , n_layer= options['num_layers']
                   , dataset_name= options['dataset_name']
                   )
    trainer = Trainer(Model, options)
    trainer.start_train()

    Model = MidLog(model_name='gru'
                   , n_model= 4
                   , d_embd= 64
                   , n_events=options['num_classes']
                   , hidden_size=options['hidden_size']
                   , n_layer=options['num_layers']
                   , dataset_name= options['dataset_name']
                   )

    options['model_path'] = "../results/multi_head/midlog_bestloss.pth"
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


def predict():

    Model = MidLog(model_name='gru'
                   , n_model= 4
                   , d_embd= 64
                   , n_events=options['num_classes']
                   , hidden_size = options['hidden_size']
                   , n_layer= options['num_layers']
                   )

    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', choices=['train', 'predict'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    else:
        predict()
