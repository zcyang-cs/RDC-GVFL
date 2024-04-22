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
    """
        # data_dict(torch_geometric)数据格式
        'dataset': dataset,                         # Dataset(torch_geometric)
        'data': data,                               # Data(torch_geometric)
        'features': features,                       # tensor
        'labels': labels,                           # tensor
        'edge_index': edge_index,                   # tensor
        'adj': adj,                                 # scipy.sparse._csr.csr_matrix
        'train_idx': train_idx,                     # tensor
        'val_idx': val_idx,                         # tensor
        'test_idx': test_idx                        # tensor

        # data_dict(deeprobust)数据格式
        'dataset': dataset,                         # Dataset(DeepRobust)
        'data': dataset,                            # Dataset(DeepRobust)
        'features': features,                       # numpy.ndarray                 -> torch.sparse_tensor
        'adj': adj,                                 # scipy.sparse._csr.csr_matrix  -> torch.sparse_tensor
        'labels': labels,                           # numpy.ndarray                 -> torch.tensor
        'idx_train': train_idx,                     # numpy.ndarray                 -> torch.tensor
        'idx_val': val_idx,                         # numpy.ndarray                 -> torch.tensor
        'idx_test': test_idx                        # numpy.ndarray                 -> torch.tensor
    """

    if para_dict['dataset'] in ['Cora', 'Citeseer', 'Pubmed']:
        dataset_name = para_dict['dataset']
        # method1: torch_geometric加载数据
        # dataset = Planetoid(root='../data/raw_data', name='Cora')
        # data = dataset[0]
        #
        # features = data.x                                   # features: (2708, 1433)
        # labels = data.y                                     # labels: (2708)
        # edge_index = data.edge_index                        # data.edge_index (2, 10556)
        # adj = to_scipy_sparse_matrix(edge_index).tocsr()    # adj: (2708, 2708) coo -> csr
        #
        # train_idx = torch.argwhere(data.train_mask).squeeze()
        # val_idx = torch.argwhere(data.val_mask).squeeze()
        # test_idx = torch.argwhere(data.test_mask).squeeze()

        # method2: deeprobust加载数据
        dataset = Planetoid(root='../data/raw_data', name=dataset_name)
        Dpr_dataset = Pyg2Dpr(dataset)
        Dpr_data = Dpr_dataset

        features = Dpr_data.features
        labels = Dpr_data.labels
        adj = Dpr_data.adj
        train_idx, val_idx, test_idx = Dpr_data.idx_train, Dpr_data.idx_val, Dpr_data.idx_test

        data_dict = {
            # data_dict(torch_geometric)数据格式
            # 'dataset': dataset,                     # Dataset(torch_geometric)
            # 'data': data,                           # Data(torch_geometric)
            # 'features': features,                   # tensor
            # 'labels': labels,                       # tensor
            # 'edge_index': edge_index,               # tensor
            # 'adj': adj,                             # scipy.sparse._csr.csr_matrix
            # 'train_idx': train_idx,                 # tensor
            # 'val_idx': val_idx,                     # tensor
            # 'test_idx': test_idx                    # tensor

            # data_dict(deeprobust)数据格式, 非tensor通过utils.to_tensor转成tensor(稀疏变量->稀疏张量,普通变量->普通张量)
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
        # csr存储的 ['indices', 'indptr', 'format', 'shape', 'data']
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
            # data_dict(deeprobust)数据格式, 非tensor通过utils.to_tensor转成tensor(稀疏变量->稀疏张量,普通变量->普通张量)
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
    # todo: 定义其他的划分方式,不同的划分方式也会对应不同的log;建立不同数据集和log的对应关系
    client_num = para_dict['num_clients']
    features = data_dict['features']

    remainder = features.shape[1] % client_num
    features_list = np.array_split(features[..., :features.shape[1] - remainder], client_num, axis=1)
    # features_list[-1] = torch.cat([feature_list[-1], features[..., -remainder:]], dim=1)

    data_dict['features_list'] = features_list

    return data_dict


