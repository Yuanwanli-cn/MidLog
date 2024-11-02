import gc
import os
import sys
import time
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.multi_head.log import log_dataset,experience_replay
from utils.multi_head.sample import sliding_window, session_window
from collections import Counter
from torch.nn import functional as F
import numpy as np
import utils.data_utils.save as save

def generate(name,win_size = 10):
    window_size = win_size
    dataset = {}
    length = 0
    with open('../data/multiHeadRNN/' + name, 'r') as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            dataset[tuple(ln)] = dataset.get(tuple(ln), 0) + 1
            length += 1
    print('Number of sessions({}): {}'.format(name, len(dataset)))
    return dataset, length

class trainer:
    def __init__(self, model, options):
        self.model_name = options['model_name']
        self.save_dir = options['save_dir']
        self.data_dir = options['data_dir']
        self.window_size = options['window_size']
        self.batch_size = options['batch_size']

        self.device = options['device']

        self.lr_step = options['lr_step']
        self.lr_decay_ratio = options['lr_decay_ratio']
        self.accumulation_step = options['accumulation_step']
        self.max_epoch = options['max_epoch']
        self.train_datas = options['train_data']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.sample = options['sample']
        self.feature_num = options['feature_num']
        self.train_loader, self.valid_loader, self.num_train_log, self.num_valid_log = self._build_loader()
        self.start_epoch = 0
        self.best_loss = 1e10
        self.best_loss1 = 1e10
        self.best_score = -1
        # self.train_model = rnn_model.to(self.device)
        self.model = model.to(self.device)
        self.optimizer = self.init_optimizer(options['optimizer'], options['lr'])
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min',patience=5,factor=0.5,cooldown=5,min_lr=options['lr']*0.05)
        if options['resume_path'] is not None:
            self.resume(options['resume_path'])
        self.save_parameters(options, self.save_dir + "parameters.txt")

        self.record = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

    def _valid(self):
        self.model.eval()
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        # tbar = tqdm(self.valid_loader, desc="\r")
        num_batch = len(self.valid_loader)
        for i, (log, label) in enumerate(self.valid_loader):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(self.device))
                # output = self.model(x=features[0], device=self.device)
                output = self.model(x=features[0], device=self.device)
                loss = criterion(output, label.to(self.device))
                total_losses += float(loss)
        # print("Validation loss:", total_losses / num_batch)
        return total_losses / num_batch

    def _train(self):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        num_batch = len(self.train_loader)
        total_losses = 0
        for i, (log, label) in enumerate(self.train_loader):
            features = list()
            for value in log.values():
                features.append(value.clone().detach().to(self.device))
            # output = self.model(features[0])
            output = self.model(features[0])
            loss = criterion(output, label.to(self.device))
            total_losses += float(loss)
            loss /= self.accumulation_step
            if (i + 1) % self.accumulation_step == 0:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            # tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))
        return total_losses / num_batch

    def _build_loader(self):
        if self.sample == 'sliding_window':
            train_logs, train_labels = sliding_window(self.data_dir,
                                                      datatype='train',
                                                      window_size=self.window_size,
                                                      sample_ratio = 0.6,
                                                      data_names= self.train_datas,

                                                      )
            val_logs, val_labels = sliding_window(self.data_dir,
                                                  datatype='val',
                                                  window_size=self.window_size,
                                                  sample_ratio=0.01,
                                                  data_names = self.train_datas
                                                  )
        elif self.sample == 'session_window':
            train_logs, train_labels = session_window(self.data_dir,
                                                      datatype='train')
            val_logs, val_labels = session_window(self.data_dir,
                                                  datatype='val')
        else:
            raise NotImplementedError

        train_dataset = log_dataset(logs=train_logs,
                                    labels=train_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)
        valid_dataset = log_dataset(logs=val_logs,
                                    labels=val_labels,
                                    seq=self.sequentials,
                                    quan=self.quantitatives,
                                    sem=self.semantics)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=True,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  pin_memory=True)
        num_train_log = len(train_dataset)
        num_valid_log = len(valid_dataset)

        self._remove_space(train_dataset, valid_dataset)
        return train_loader, valid_loader, num_train_log, num_valid_log

    def _remove_space(self, *params):
        for param in params:
            del param
        gc.collect()

    def init_optimizer(self, opti_name, learning_rate):
        if opti_name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=learning_rate,
                                        momentum=0.9)
        elif opti_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
            )
        elif opti_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                self.model.parameters(),
                lr=learning_rate,
                alpha = 0.9,
            )
        else:
            raise NotImplementedError
        return optimizer

    def resume(self, path, load_optimizer=True):
        print("Resuming from {}".format(path))
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.record = checkpoint['record']
        self.best_score = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys() and load_optimizer:
            print("Loading optimizer state dict")
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def add_record(self, outer_key, inner_dict):
        for inner_key, value in inner_dict.items():
            self.record[outer_key][inner_key].append(value)

    def save_parameters(self, options, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w+") as f:
            for key in options.keys():
                f.write("{}: {}\n".format(key, options[key]))

    def save_checkpoint(self, epoch, save_optimizer=True, suffix=""):
        checkpoint = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "best_loss": self.best_loss,
            "record": self.record,
            "best_score": self.best_score
        }
        if save_optimizer:
            checkpoint['optimizer'] = self.optimizer.state_dict()
        save_path = self.save_dir + self.model_name + "_" + suffix + ".pth"
        torch.save(checkpoint, save_path)
        # print("Save rnn_model checkpoint at {}".format(save_path))

    def start_train1(self):
        inner_dict = dict()
        msg = 'Starting epoch: {0:<5d}| phase:{1:<10}|⏰: {2:<10}| Learning rate: {3:.5f} | {1}loss: {4}'
        last_best_epoch = 0
        break_flag = False
        for epoch in range(self.start_epoch, self.max_epoch):
            if epoch == 0:
                self.optimizer.param_groups[0]['lr'] /= 32
            if epoch in [1, 2, 3, 4, 5]:
                self.optimizer.param_groups[0]['lr'] *= 2
            if epoch in self.lr_step:
                self.optimizer.param_groups[0]['lr'] *= self.lr_decay_ratio

            #train
            inner_dict['time'] = time.strftime("%H:%M:%S")
            inner_dict['loss'] = self._train()
            inner_dict['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            inner_dict['epoch'] = epoch
            print(msg.format(inner_dict['epoch'], 'train', inner_dict['time'], inner_dict['lr'],inner_dict['loss']))
            self.add_record(outer_key='train', inner_dict=inner_dict)

            # valid
            if epoch >= self.max_epoch/2 and epoch % 2 == 0:
            # if epoch % 2 == 0:
                inner_dict['loss'] = self._valid()
                inner_dict['time'] = time.strftime("%H:%M:%S")
                print(msg.format(inner_dict['epoch'], 'valid', inner_dict['time'], inner_dict['lr'],inner_dict['loss'])+'\n')
                self.add_record(outer_key='train', inner_dict=inner_dict)
                self.save_checkpoint(epoch,
                                     save_optimizer=False,
                                     suffix="epoch" + str(epoch))
                if (epoch-last_best_epoch< 100):
                    if (inner_dict['loss'] < self.best_loss):
                        last_best_epoch = epoch
                        self.best_loss = inner_dict['loss'] if (inner_dict['loss'] < self.best_loss) else self.best_loss
                        self.save_checkpoint(epoch,save_optimizer=True,suffix="bestloss")
                else:
                    break_flag = True
            if break_flag:
                self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
                break
            elif epoch == self.max_epoch-1:
                self.save_checkpoint(epoch, save_optimizer=True, suffix="last")

    def start_train(self):
        inner_dict = dict()
        valid_data,train_data = [],[]
        msg = 'Starting epoch: {0:<5d}| phase:{1:<10}|⏰: {2:<10}| Learning rate: {3:.5f} | {1}loss: {4}'
        for epoch in range(self.start_epoch, self.max_epoch):
            # train
            train_loss = self._train()
            inner_dict['time'] = time.strftime("%H:%M:%S")
            inner_dict['loss'] = train_loss
            inner_dict['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            inner_dict['epoch'] = epoch
            print(msg.format(inner_dict['epoch'], 'train', inner_dict['time'], inner_dict['lr'], inner_dict['loss']))
            # 添加记录
            train_data.append([epoch,train_loss])
            self.add_record(outer_key='train', inner_dict=inner_dict)
            # valid
            # if epoch > self.max_epoch*0.2 and epoch % 2 == 0:
            if epoch % 2 == 0:
                valid_loss = self._valid()
                self.lr_scheduler.step(valid_loss)
                valid_data.append([epoch, valid_loss])
                inner_dict['loss'] = valid_loss
                inner_dict['time'] = time.strftime("%H:%M:%S")
                print(msg.format(inner_dict['epoch'], 'valid', inner_dict['time'], inner_dict['lr'],inner_dict['loss'])+'\n')
                self.add_record(outer_key='train', inner_dict=inner_dict)
                self.save_checkpoint(epoch,
                                     save_optimizer=False,
                                     suffix="epoch" + str(epoch))
                if inner_dict['loss'] < self.best_loss:
                    self.best_loss = inner_dict['loss']
                    self.save_checkpoint(epoch,save_optimizer=False,suffix="bestloss")
            self.save_checkpoint(epoch, save_optimizer=True, suffix="last")
        # 保存train loss和valid loss
        save.save_excel(self.save_dir + 'train_loss.xls', train_data, 'epoch', 'loss')
        save.save_excel(self.save_dir + 'valid_loss.xls', valid_data, 'epoch', 'loss')
    def save_record(self):
        try:
            for key, values in self.record.items():
                pd.DataFrame(values).to_csv(self.save_dir + key + "_log.csv",
                                            index=False)
            print("Record saved")
        except:
            print("Failed to save record")


class predicter:
    def __init__(self, model, options):
        self.data_dir = options['data_dir']
        self.device = options['device']
        self.model = model
        self.model_path = options['model_path']
        self.window_size = options['window_size']
        self.num_candidates = options['num_candidates']
        self.num_classes = options['num_classes']
        self.input_size = options['input_size']
        self.sequentials = options['sequentials']
        self.quantitatives = options['quantitatives']
        self.semantics = options['semantics']
        self.batch_size = options['batch_size']
        self.detected_sequences = experience_replay()

    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate('hdfs_test_normal',win_size=self.window_size)

        test_abnormal_loader, test_abnormal_length = generate('hdfs_test_abnormal',win_size=self.window_size)
        TP = 0
        FP = 0
        # Test the rnn_model
        start_time = time.time()
        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    is_exist, find_result = self.detected_sequences.find_sequence(seq0)
                    if not is_exist:
                        seq1 = [0] * 28
                        log_conuter = Counter(seq0)
                        for key in log_conuter:
                            seq1[key] = log_conuter[key]

                        seq0 = torch.tensor(seq0, dtype=torch.float).view(
                            -1, self.window_size, self.input_size).to(self.device)
                        label = torch.tensor(label).view(-1).to(self.device)
                        output = model(x=seq0, device=self.device)
                        # self.roc_model.append_data_v1(output, label,self.num_candidates)
                        # self.roc_model.append_data(output, label)
                        predicted = torch.argsort(output,1)[0][-self.num_candidates:]
                        self.detected_sequences.add_sequence(seq0,predicted)
                        if label not in predicted:
                            FP += test_normal_loader[line]
                            break
                    else:
                        predicted = find_result
                        if label not in predicted:
                            FP += test_normal_loader[line]
                            break

        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]
                    # seq1 = torch.tensor(seq1, dtype=torch.float).view(
                    #     -1, self.num_classes, self.input_size).to(self.device)
                    if -1 not in seq0:
                        is_exist, find_result = self.detected_sequences.find_sequence(seq0)
                        if not is_exist:
                            seq0 = torch.tensor(seq0, dtype=torch.float).view(-1, self.window_size, self.input_size).to(self.device)
                            label = torch.tensor(label).view(-1).to(self.device)
                            output = model(x=seq0, device=self.device)
                            # self.roc_model.append_data_v1(output, label,self.num_candidates)
                            # self.roc_model.append_data(output, label)
                            predicted = torch.argsort(output,1)[0][-self.num_candidates:]
                            # self.detected_sequences.add_sequence(seq0, predicted)
                            if label not in predicted:
                                TP += test_abnormal_loader[line]
                                break
                        else:
                            predicted = find_result
                            if label not in predicted:
                                TP += test_abnormal_loader[line]
                                break
                    else:
                        TP += test_abnormal_loader[line]
                        break

        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        print('experience pool length:', len(self.detected_sequences.experience_pool.keys()))
        return elapsed_time,FP, FN, P, R, F1

    def predict_unsupervised1(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_normal_loader, test_normal_length = generate('bgl_test_normal')

        test_abnormal_loader, test_abnormal_length = generate(
            'bgl_test_abnormal')

        TP = 0
        FP = 0
        # Test the rnn_model
        start_time = time.time()
        with torch.no_grad():
            for line in tqdm(test_normal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    label = torch.tensor(label).view(-1).to(self.device)
                    output = model(x=seq0, device=self.device)
                    # self.roc_model.append_data_v1(output, label,self.num_candidates)
                    # self.roc_model.append_data(output, label)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]
                    if label not in predicted:
                        FP += test_normal_loader[line]
                        break

        with torch.no_grad():
            for line in tqdm(test_abnormal_loader.keys()):
                for i in range(len(line) - self.window_size):
                    seq0 = line[i:i + self.window_size]
                    label = line[i + self.window_size]
                    seq1 = [0] * 28
                    log_conuter = Counter(seq0)
                    for key in log_conuter:
                        seq1[key] = log_conuter[key]

                    seq0 = torch.tensor(seq0, dtype=torch.float).view(
                        -1, self.window_size, self.input_size).to(self.device)
                    # seq1 = torch.tensor(seq1, dtype=torch.float).view(
                    #     -1, self.num_classes, self.input_size).to(self.device)
                    if -1 not in seq0:
                        label = torch.tensor(label).view(-1).to(self.device)

                        output = model(x=seq0, device=self.device)
                        # self.roc_model.append_data_v1(output, label,self.num_candidates)
                        # self.roc_model.append_data(output, label)
                        predicted = torch.argsort(output,1)[0][-self.num_candidates:]
                        if label not in predicted:
                            TP += test_abnormal_loader[line]
                            break
                    else:
                        TP += test_abnormal_loader[line]
                        break

        # Compute precision, recall and F1-measure
        FN = test_abnormal_length - TP
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(FP, FN, P, R, F1))
        print('Finished Predicting')
        elapsed_time = time.time() - start_time
        print('elapsed_time: {}'.format(elapsed_time))
        return elapsed_time, FP, FN, P, R, F1

    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)['state_dict'])
        model.eval()
        print('model_path: {}'.format(self.model_path))
        test_logs, test_labels = session_window(self.data_dir, datatype='test')
        test_dataset = log_dataset(logs=test_logs,
                                   labels=test_labels,
                                   seq=self.sequentials,
                                   quan=self.quantitatives,
                                   sem=self.semantics)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      pin_memory=True)
        tbar = tqdm(self.test_loader, desc="\r")
        TP, FP, FN, TN = 0, 0, 0, 0
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)
            output = F.sigmoid(output)[:, 0].cpu().detach().numpy()
            # predicted = torch.argmax(output, dim=1).cpu().numpy()
            predicted = (output < 0.2).astype(int)
            label = np.array([y.cpu() for y in label])
            TP += ((predicted == 1) * (label == 1)).sum()
            FP += ((predicted == 1) * (label == 0)).sum()
            FN += ((predicted == 0) * (label == 1)).sum()
            TN += ((predicted == 0) * (label == 0)).sum()
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(
            'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
                .format(FP, FN, P, R, F1))