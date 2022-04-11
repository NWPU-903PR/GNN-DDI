# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:07:17 2021

@author: A
"""

from datetime import datetime
import time
import argparse
import torch.nn.functional as F

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
import json
import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader, split_dataset

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--test_size', type=float, default=0.2, help='test size of samples')
parser.add_argument('--val_size', type=float, default=0.05, help='val size of samples')
parser.add_argument('--out_dim', type=int, default=128, help='dim of drug features output from encoder')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=10, help='num of epochs')
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])

args = parser.parse_args()
test_size = args.test_size
val_size = args.val_size
out_dim = args.out_dim
lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################

###### Dataset

filename = '/home/disk2/fengyuehua/PycharmProjects/paper4/data/adj.csv'

adj = pd.read_csv(filename, index_col=0, header=0, names=[i for i in range(1935)])  # 这里不重新指定就变成字符串
#
# fpath = '/home/disk2/fengyuehua/PycharmProjects/paper4/data/'
# drug_features = json.load(open(fpath + "drug_graph.json"))
# drug_features = pd.DataFrame(drug_features)
train_edge, test_edge, val_edge = split_dataset(adj, test_size, val_size)
train_type = np.ones(len(train_edge))
test_type = np.ones(len(test_edge))
val_type = np.ones(len(val_edge))

train_tup = [(h, t, r) for h, t, r in zip(train_edge[:, 0], train_edge[:, 1], train_type)]
test_tup = [(h, t, r) for h, t, r in zip(test_edge[:, 0], test_edge[:, 1], test_type)]
val_tup = [(h, t, r) for h, t, r in zip(val_edge[:, 0], val_edge[:, 1], val_type)]

# train_data = DrugDataset(train_edge,drug_features,ratio=data_size_ratio, neg_ent=neg_samples)
# val_data = DrugDataset(val_edge, drug_features, ratio=data_size_ratio)
# test_data = DrugDataset(test_edge, drug_features)

# df_ddi_train = pd.read_csv('data/ddi_training.csv')
# df_ddi_val = pd.read_csv('data/ddi_validation.csv')
# df_ddi_test = pd.read_csv('data/ddi_test.csv')


# train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
# val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'])]
# test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]
#
# train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
# val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
# test_data = DrugDataset(test_tup, disjoint_split=False)

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio)
test_data = DrugDataset(test_tup)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size * 3)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size * 3)


def do_compute(batch, device, training=True):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    pos_edges, neg_edges = batch

    pos_edges = [tensor.to(device=device) for tensor in pos_edges]
    p_score = model(pos_edges)  ###model 的输出是预测的分数
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_edges = [tensor.to(device=device) for tensor in neg_edges]
    n_score = model(neg_edges)  ###model 的输出是预测的分数
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())

    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred_l = np.argmax(probas_pred, axis=1)  ###.detach().numpy()
    test_y = target
    pred = probas_pred  ##pred_p.detach().numpy()
    # pred = pred[:,1]

    ###================================
    precision, recall, pr_thresholds = metrics.precision_recall_curve(test_y, pred)
    auprc_score = metrics.auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = metrics.roc_curve(test_y, pred)
    auc_score = metrics.auc(fpr, tpr)
    predicted_score = np.zeros(shape=(len(test_y), 1))
    predicted_score[pred > threshold] = 1
    confusion_matri = metrics.confusion_matrix(y_true=test_y, y_pred=predicted_score)
    # print("confusion_matrix:", confusion_matri)
    f1 = metrics.f1_score(test_y, predicted_score)
    accuracy = metrics.accuracy_score(test_y, predicted_score)
    precision = metrics.precision_score(test_y, predicted_score)
    recall = metrics.recall_score(test_y, predicted_score)
    print("new auc_score:", auc_score)
    print('new accuracy:', accuracy)
    print("new precision:", precision)
    print("new recall:", recall)
    print("new f1:", f1)
    print("new auprc_score:", auprc_score)
    print("===================")

    #
    # pred = (probas_pred >= 0.5).astype(np.int)
    #
    # acc = metrics.accuracy_score(target, pred)
    # auc_roc = metrics.roc_auc_score(target, probas_pred)
    # f1_score = metrics.f1_score(target, pred)
    #
    # p, r, t = metrics.precision_recall_curve(target, probas_pred)
    # auc_prc = metrics.auc(r, p)

    return accuracy, auc_score, auprc_score


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    print(model)

    for i in range(1, n_epochs + 1):
        start = time.time()
        train_loss = 0
        val_loss = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            ground_truth = torch.tensor(ground_truth, dtype=torch.float).to(device=device)
            # probas_pred = torch.tensor(np.hstack(probas_pred), dtype = torch.float).to(device=device)
            score = torch.cat([p_score, n_score], 0).squeeze()

            loss = F.binary_cross_entropy_with_logits(score, ground_truth)
            # F.binary_cross_entropy_with_logits()
            # print(probas_pred.grad)
            # print(ground_truth.grad)
            # print("probas_pred:{}".format(p_score.requires_grad))
            # print("ground_truth:{}".format(n_score.grad))
            # print("loss:{}".format(loss.requires_grad))
            # loss, loss_p, loss_n = loss_fn(p_score, n_score)  ###weighted loss strategy

            optimizer.zero_grad()
            # loss.requires_grad_(True)
            loss.backward()
            print(loss)
            optimizer.step()

        train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_auc_prc = do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                score = torch.cat([p_score, n_score], 0).squeeze()

                ground_truth = torch.tensor(ground_truth, dtype=torch.float).to(device=device)
                # probas_pred = torch.tensor(np.hstack(probas_pred), dtype=torch.float).to(device=device)

                loss = F.binary_cross_entropy(score, ground_truth)
                # loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)
                # print(probas_pred.grad)
                # print(ground_truth.grad)
                # print("probas_pred:{}".format(p_score.requires_grad))
                # print("ground_truth:{}".format(n_score.requires_grad))
                # print("loss:{}".format(loss.requires_grad))

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_auc_prc = do_compute_metrics(val_probas_pred, val_ground_truth)

        if scheduler:
            # print('scheduling')
            scheduler.step()

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
              f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(
            f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_auprc: {train_auc_prc:.4f}, val_auprc: {val_auc_prc:.4f}')


###n_atom_feats 就是药物的结构特征 n
model = model_binary.DDI(drug_features, num_features_xd=78, out_dim=64, dropoutc=0.1)
loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
# print(model)
model.to(device=device)

# if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, scheduler)


