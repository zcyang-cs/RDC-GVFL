import torch
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_scipy_sparse_matrix
from deeprobust.graph.data import Pyg2Dpr
import numpy as np
from copy import deepcopy
from matplotlib import pyplot
import scipy.sparse as sp


def load_data(para_dict):
    if para_dict['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset_name = para_dict['dataset']
        dataset = Planetoid(root='../data/raw_data', name=dataset_name)
        Dpr_dataset = Pyg2Dpr(dataset)
        Dpr_data = Dpr_dataset

        features = Dpr_data.features
        labels = Dpr_data.labels
        adj = Dpr_data.adj
        train_idx, val_idx, test_idx = Dpr_data.idx_train, Dpr_data.idx_val, Dpr_data.idx_test

        data_dict = {
            'dataset': Dpr_dataset,                     # Dataset(deeprobust)
            'data': Dpr_data,                           # Dataset(deeprobust)
            'features': features,                       # numpy.ndarray
            'labels': labels,                           # numpy.ndarray
            'adj': adj,                                 # scipy.sparse._csr.csr_matrix
            'train_idx': train_idx,                     # numpy.ndarray
            'val_idx': val_idx,                         # numpy.ndarray
            'test_idx': test_idx                        # numpy.ndarray
        }

        return data_dict

    if para_dict['dataset'] == 'Cora_ML':
        adj_loader = np.load("../data/raw_data/Cora_ML/raw/cora_ml_adj.npz")
        adj = sp.csr_matrix((adj_loader['data'], adj_loader['indices'],
                             adj_loader['indptr']), shape=adj_loader['shape'])

        feature_loader = np.load("../data/raw_data/Cora_ML/raw/cora_ml_features.npz")
        features = sp.csr_matrix((feature_loader['data'], feature_loader['indices'],
                                  feature_loader['indptr']), shape=feature_loader['shape'])
        features = features.todense().A

        labels = np.load("../data/raw_data/Cora_ML/raw/cora_ml_label.npy")
        train_idx = np.load("../data/raw_data/Cora_ML/raw/cora_ml_train_node.npy")
        val_idx = np.load("../data/raw_data/Cora_ML/raw/cora_ml_val_node.npy")
        test_idx = np.load("../data/raw_data/Cora_ML/raw/cora_ml_test_node.npy")

        Dpr_dataset = 'Cora_ML'
        Dpr_data = 'Cora_ML'

        data_dict = {
            'dataset': Dpr_dataset,         # Dataset(deeprobust)
            'data': Dpr_data,               # Dataset(deeprobust)
            'features': features,           # numpy.ndarray
            'labels': labels,               # numpy.ndarray
            'adj': adj,                     # scipy.sparse._csr.csr_matrix
            'train_idx': train_idx,         # numpy.ndarray
            'val_idx': val_idx,             # numpy.ndarray
            'test_idx': test_idx            # numpy.ndarray
        }

        return data_dict


def partition(para_dict, data_dict):
    client_num = para_dict['num_clients']
    features = data_dict['features']

    remainder = features.shape[1] % client_num
    features_list = np.array_split(features[..., :features.shape[1] - remainder], client_num, axis=1)
    # features_list[-1] = torch.cat([feature_list[-1], features[..., -remainder:]], dim=1)

    data_dict['features_list'] = features_list

    return data_dict
