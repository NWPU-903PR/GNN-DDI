# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 21:49:27 2021

@author: A
"""
import itertools
from collections import defaultdict
from operator import neg
import random
import math
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np


"""

df_drugs_smiles = pd.read_csv('data/drug_smiles.csv')

DRUG_TO_INDX_DICT = {drug_id: indx for indx, drug_id in enumerate(df_drugs_smiles['drug_id'])}

drug_id_mol_graph_tup = [(id, Chem.MolFromSmiles(smiles.strip())) for id, smiles in zip(df_drugs_smiles['drug_id'], df_drugs_smiles['smiles'])]


# Gettings information and features of atoms
ATOM_MAX_NUM = np.max([m[1].GetNumAtoms() for m in drug_id_mol_graph_tup])
AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)})
max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup))
max_valence = max(max_valence, 9)
AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_ATOM_FC = MAX_ATOM_FC if MAX_ATOM_FC else 0
MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m[1].GetAtoms() for m in drug_id_mol_graph_tup)]))
MAX_RADICAL_ELC = MAX_RADICAL_ELC if MAX_RADICAL_ELC else 0


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_atom_features(atom, mode='one_hot'):

    if mode == 'one_hot':
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            one_of_k_encoding_unk(atom.GetDegree(), AVAILABLE_ATOM_DEGREES),
            one_of_k_encoding_unk(atom.GetTotalNumHs(), AVAILABLE_ATOM_TOTAL_HS),
            one_of_k_encoding_unk(atom.GetImplicitValence(), AVAILABLE_ATOM_VALENCE),
            torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
        ])
    else:
        atom_feature = torch.cat([
            one_of_k_encoding_unk(atom.GetSymbol(), AVAILABLE_ATOM_SYMBOLS),
            torch.tensor([atom.GetDegree()]).float(),
            torch.tensor([atom.GetTotalNumHs()]).float(),
            torch.tensor([atom.GetImplicitValence()]).float(),
            torch.tensor([atom.GetIsAromatic()]).float()
        ])

    return atom_feature


def get_mol_edge_list_and_feat_mtx(mol_graph):
    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    
    # assert TOTAL_ATOM_FEATS == features.shape[-1], "Expected atom n_features and retrived n_features not matching"
    return undirected_edge_list.T, features


MOL_EDGE_LIST_FEAT_MTX = {drug_id: get_mol_edge_list_and_feat_mtx(mol) 
                                for drug_id, mol in drug_id_mol_graph_tup}
MOL_EDGE_LIST_FEAT_MTX = {drug_id: mol for drug_id, mol in MOL_EDGE_LIST_FEAT_MTX.items() if mol is not None}

TOTAL_ATOM_FEATS = (next(iter(MOL_EDGE_LIST_FEAT_MTX.values()))[1].shape[-1])
"""

##### DDI statistics and counting #######

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

#df_all_pos_ddi = pd.read_csv('data/ddis.csv')
#all_pos_tup = [(h, t, r) for h, t, r in zip(df_all_pos_ddi['d1'], df_all_pos_ddi['d2'], df_all_pos_ddi['type'])]


filename =  '/home/disk2/fengyuehua/PycharmProjects/paper4/data/adj.csv'
adj = pd.read_csv(filename,index_col=0, header=0,names=[i for i in range(1935)])  # 这里不重新指定就变成字符串
adj = np.array(adj)
adj = np.triu(adj)
adj = pd.DataFrame(adj)

adj_mats = sp.csr_matrix(adj)
edges_all, _, _ = sparse_to_tuple(adj_mats)




#ALL_DRUG_IDS, _ = zip(*drug_id_mol_graph_tup)
#ALL_DRUG_IDS = np.array(list(set(ALL_DRUG_IDS)))


ALL_TRUE_H_WITH_TR = defaultdict(list)
ALL_TRUE_T_WITH_HR = defaultdict(list)


FREQ_REL = len(edges_all)
ALL_H_WITH_R = defaultdict(dict)
ALL_T_WITH_R = defaultdict(dict)
ALL_TAIL_PER_HEAD = {}
ALL_HEAD_PER_TAIL = {}

for h, t in edges_all:
    ALL_TRUE_H_WITH_TR[t].append(h)
    ALL_TRUE_T_WITH_HR[h].append(t)
    ALL_H_WITH_R[h] = 1
    ALL_T_WITH_R[t] = 1

for t in ALL_TRUE_H_WITH_TR:
    ALL_TRUE_H_WITH_TR[t] = np.array(list(set(ALL_TRUE_H_WITH_TR[t])))
for h in ALL_TRUE_T_WITH_HR:
    ALL_TRUE_T_WITH_HR[h] = np.array(list(set(ALL_TRUE_T_WITH_HR[h])))

ALL_H_WITH_R = np.array(list(ALL_H_WITH_R.keys()))
ALL_T_WITH_R = np.array(list(ALL_T_WITH_R.keys()))
ALL_HEAD_PER_TAIL = FREQ_REL / len(ALL_T_WITH_R)
ALL_TAIL_PER_HEAD = FREQ_REL / len(ALL_H_WITH_R)

#######    ****** ###############

#class TestbedDataset(Dataset):
#    def __init__(self, drug_id, drug_features):
#        super(TestbedDataset, self).__init__()
#        self.drug_id = drug_id
#        self.drug_features = drug_features
#
#
#    def len(self):
#        return len(self.drug_id)
#
#    def get(self,idx):
#
#        #drug_feature
#        id = self.drug_id[idx]
#        # print(id)
#        c_size=self.drug_features.loc[ 'c_size',str(id)]
#        features=self.drug_features.loc['features',str(id)]
#        features = torch.tensor(features[0],dtype=torch.float)
#        #print(features1.size())
#        edge_index=self.drug_features.loc['edge_index',str(id)]
#        edge_index = torch.tensor(edge_index[0],dtype=torch.long)
#        #print(edge_index1.size())
#        # make the graph ready for PyTorch Geometrics GCN algorithms:
#        Data = DATA.Data(x=features,edge_index=edge_index.transpose(1, 0))
#        # Data.y = torch.tensor([float(syn)], dtype=torch.float)  # regress
#
#        Data.__setitem__('c_size', torch.tensor([c_size],dtype=torch.long))
#        return Data

ALL_DRUG_IDS = np.arange(1935)

class DrugDataset(Dataset):
    def __init__(self, edges, drug_features,ratio=1.0,  neg_ent=1):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        ''' 
        self.neg_ent = neg_ent
        self.edges = edges
        self.ratio = ratio
        
        self.drug_features = drug_features
        self.drug_ids = ALL_DRUG_IDS


    def __len__(self):
        return len(self.edges)
    
    def __getitem__(self, index):
        return self.edges[index]

    def collate_fn(self, batch):  ###将（head，tail，relation）三元组转成（head节点data（包含节点分子图edge_index,feature),
                                  ### tail节点data（包含节点分子图edge_index,feature), relation)三元组
        pos_h_samples = []
        pos_t_samples = []
        neg_h_samples = []
        neg_t_samples = []

        for h, t in batch:
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)
            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)

            neg_heads, neg_tails = self.__normal_batch(h, t, self.neg_ent)   ####生成负样本

            for neg_h in neg_heads:
                neg_h_samples.append(self.__create_graph_data(neg_h))
                neg_t_samples.append(t_data)

            for neg_t in neg_tails:
                neg_h_samples.append(h_data)
                neg_t_samples.append(self.__create_graph_data(neg_t))

        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_tri = (pos_h_samples, pos_t_samples)

        neg_h_samples = Batch.from_data_list(neg_h_samples)
        neg_t_samples = Batch.from_data_list(neg_t_samples)
        neg_tri = (neg_h_samples, neg_t_samples)

        return pos_tri, neg_tri



    
            
    def __create_graph_data(self, id):
        #drug_feature
        id = self.drug_ids[id]
        # print(id)
        c_size=self.drug_features.loc[ 'c_size',str(id)]
        features=self.drug_features.loc['features',str(id)]
        features = torch.tensor(features[0],dtype=torch.float)
        #print(features1.size())
        edge_index=self.drug_features.loc['edge_index',str(id)]
        edge_index = torch.tensor(edge_index[0],dtype=torch.long)
        #print(edge_index1.size())
        # make the graph ready for PyTorch Geometrics GCN algorithms:
        drugData = Data(x=features,edge_index=edge_index.transpose(1, 0))
        # Data.y = torch.tensor([float(syn)], dtype=torch.float)  # regress
#        drugData.__setitem__('c_size', torch.tensor([c_size],dtype=torch.long))

        
        return drugData
    

    def __corrupt_ent(self, drug, true_DDI_dict, max_num=1):
        corrupted_ents = []
        current_size = 0
        while current_size < max_num:
            candidates = np.random.choice(self.drug_ids, (max_num - current_size) * 2)  ###任意选择两个
            mask = np.isin(candidates, true_DDI_dict[drug], assume_unique=True, invert=True)
            corrupted_ents.append(candidates[mask])
            current_size += len(corrupted_ents[-1])
        
        if corrupted_ents != []:
            corrupted_ents = np.concatenate(corrupted_ents)

        return np.asarray(corrupted_ents[:max_num])
        
    def __corrupt_head(self, t, n=1):
        return self.__corrupt_ent(t, ALL_TRUE_H_WITH_TR, n)

    def __corrupt_tail(self, h,  n=1):
        return self.__corrupt_ent(h, ALL_TRUE_T_WITH_HR, n)
    
    def __normal_batch(self, h, t,  neg_size):
        neg_size_h = 0
        neg_size_t = 0
        # prob = self.tail_per_head[r] / (self.tail_per_head[r] + self.head_per_tail[r])
        prob = ALL_TAIL_PER_HEAD / (ALL_TAIL_PER_HEAD + ALL_HEAD_PER_TAIL)
        # prob = 2
        for i in range(neg_size):
            if random.random() < prob:
                neg_size_h += 1
            else:
                neg_size_t +=1
        
        return (self.__corrupt_head(t,  neg_size_h),
                self.__corrupt_tail(h,  neg_size_t))  


# Simple DrugDataLoaderWrapper to use the customized collate_fn
class DrugDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)
        
        
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape       

def split_dataset(adj, test_size, val_size):
   
    adj_mats = sp.csr_matrix(adj)
    edges_all, _, _ = sparse_to_tuple(adj_mats)
    num_test = max(50, int (np.floor(edges_all.shape[0] * test_size)))
    num_val = max(50, int(np.floor(edges_all.shape[0] * val_size)))

    all_edge_idx = list(range(edges_all.shape[0]))
    np.random.shuffle(all_edge_idx)

    val_edge_idx = all_edge_idx[:num_val]
    val_edges = edges_all[val_edge_idx]

    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges_all[test_edge_idx]

    train_edges = np.delete(edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0)


    return train_edges,test_edges,val_edges




