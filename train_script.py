from datetime import datetime
import time 
import argparse

import torch
from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np

import models
import custom_loss
from data_preprocessing import DrugDataset, DrugDataLoader, TOTAL_ATOM_FEATS
from data_split_cold_start import split_data

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=TOTAL_ATOM_FEATS, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=64, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=1, help='num of interaction types')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=600, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=64, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')

parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])


args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
device = 'cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################

###### Dataset
#
# df_ddi_train = pd.read_csv('data/ddi_training1.csv')
# df_ddi_val = pd.read_csv('data/ddi_validation1.csv')
# df_ddi_test = pd.read_csv('data/ddi_test1.csv')


###==== cold start============================
df_all_pos_ddi = pd.read_csv('data/ddis1.csv')

df_drugs_smiles = pd.read_csv('data/drug_smiles.csv')
all_pos_tups = [(h, t, r, index) for index, (h, t, r) in
                enumerate(zip(df_all_pos_ddi['d1'], df_all_pos_ddi['d2'], df_all_pos_ddi['type']))]
all_drugs = set(df_drugs_smiles['drug_id'])

train_tups, s1_tups, s2_tups = split_data(all_pos_tups, all_drugs)

df_ddi_train = df_all_pos_ddi.loc[[item[3] for item in train_tups]]
# df_all_pos_ddi.loc[train_tups[0][3]]
df_ddi_val = df_all_pos_ddi.loc[[item[3] for item in s1_tups]]
df_ddi_test = df_all_pos_ddi.loc[[item[3] for item in s2_tups]]
###===========cold start  end =================

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
val_tup = [(h, t, r) for h, t, r in zip(df_ddi_val['d1'], df_ddi_val['d2'], df_ddi_val['type'])]
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio)
test_data = DrugDataset(test_tup)

print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3)

def do_compute(batch, device, training=True):
        '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''
        probas_pred, ground_truth = [], []
        pos_tri, neg_tri = batch
        
        pos_tri = [tensor.to(device=device) for tensor in pos_tri]
        p_score = model(pos_tri)
        probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        ground_truth.append(np.ones(len(p_score)))

        neg_tri = [tensor.to(device=device) for tensor in neg_tri]
        n_score = model(neg_tri)
        probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        ground_truth.append(np.zeros(len(n_score)))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

        return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    #
    # pred = (probas_pred >= 0.5).astype(np.int)
    #
    # acc = metrics.accuracy_score(target, pred)
    # auc_roc = metrics.roc_auc_score(target, probas_pred)
    # f1_score = metrics.f1_score(target, pred)
    #
    # p, r, t = metrics.precision_recall_curve(target, probas_pred)
    # auc_prc = metrics.auc(r, p)

    preds_all = probas_pred
    labels_all = target

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    precision,recall,pr_thresholds = metrics.precision_recall_curve(labels_all,preds_all)
    auprc_score = metrics.auc(recall,precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = metrics.roc_curve(labels_all, preds_all)
    auc_score = metrics.auc(fpr, tpr)
    predicted_score = np.zeros(shape=(len(labels_all), 1))
    predicted_score[preds_all > threshold] = 1
    confusion_matri = metrics.confusion_matrix(y_true=labels_all, y_pred=predicted_score)
    # print("confusion_matrix:", confusion_matri)


    f = metrics.f1_score(labels_all, predicted_score)
    accuracy = metrics.accuracy_score(labels_all,predicted_score)
    precision = metrics.precision_score(labels_all,predicted_score)
    recall = metrics.recall_score(labels_all,predicted_score)



    return roc_sc, auprc_score,accuracy,precision,recall,f

    # return acc, auc_roc, auc_prc


def train(model, train_data_loader, val_data_loader, test_data_loader, loss_fn,  optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []
        train_loss_epoch = 1

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data)
        if train_loss < train_loss_epoch:
            train_loss_epoch = train_loss
            state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':i}
            torch.save(state, f= 'best.pth')

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            t_roc_sc, t_auprc_score, t_accuracy, t_precision, t_recall, t_f=  do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            v_roc_sc, v_auprc_score, v_accuracy, v_precision, v_recall, v_f = do_compute_metrics(val_probas_pred, val_ground_truth)

            # if i% 10 == 0:
            #
            #     test_loss = 0
            #     test_probas_pred = []
            #     test_ground_truth = []
            #     with torch.no_grad():
            #
            #         for batch in test_data_loader:
            #             model.eval()
            #             p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
            #             test_probas_pred.append(probas_pred)
            #             test_ground_truth.append(ground_truth)
            #             loss, loss_p, loss_n = loss_fn(p_score, n_score)
            #             test_loss += loss.item() * len(p_score)
            #
            #             test_loss /= len(test_data)
            #             test_probas_pred = np.concatenate(test_probas_pred)
            #             test_ground_truth = np.concatenate(test_ground_truth)
            #             t_roc_sc, t_auprc_score, t_accuracy, t_precision, t_recall, t_f = do_compute_metrics(
            #                 test_probas_pred,
            #                 test_ground_truth)

        if scheduler:
            # print('scheduling')
            scheduler.step()


        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},')
        print(f'\t\tv_roc_sc: {v_roc_sc:.4f}, v_auprc_score: {v_auprc_score:.4f}, v_accuracy: {v_accuracy:.4f}, v_precision: {v_precision:.4f},v_recall: {v_recall:.4f},v_f: {v_f:.4f}')

        if i%10 == 0:

            test_loss = 0
            test_probas_pred = []
            test_ground_truth = []
            with torch.no_grad():

                for batch in test_data_loader:
                    model.eval()
                    p_score, n_score, probas_pred, ground_truth = do_compute(batch, device)
                    test_probas_pred.append(probas_pred)
                    test_ground_truth.append(ground_truth)
                    loss, loss_p, loss_n = loss_fn(p_score, n_score)
                    test_loss += loss.item() * len(p_score)

                test_loss /= len(test_data)
                test_probas_pred = np.concatenate(test_probas_pred)
                test_ground_truth = np.concatenate(test_ground_truth)
                t_roc_sc, t_auprc_score, t_accuracy, t_precision, t_recall, t_f = do_compute_metrics(test_probas_pred,
                                                                                                         test_ground_truth)



            print(
                f'\t\tt_roc_sc: {t_roc_sc:.4f}, t_auprc_score: {t_auprc_score:.4f}, t_accuracy: {t_accuracy:.4f}, t_precision: {t_precision:.4f},t_recall: {t_recall:.4f},t_f: {t_f:.4f}')


model = models.SSI_DDI(n_atom_feats, n_atom_hid, kge_dim, rel_total, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2,2])
loss = custom_loss.SigmoidLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
# print(model)
model.to(device=device)

# if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, test_data_loader, loss, optimizer, n_epochs, device, scheduler)


