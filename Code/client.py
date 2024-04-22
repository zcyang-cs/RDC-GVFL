import torch
from model import GCN, SGC
from utils import utils
from flow import logger
import importlib
import scipy.sparse as sp
from utils.utils import sparse_mx_to_torch_sparse_tensor


class Client(object):
    def __init__(self, cid, para_dict, data_dict):
        self.cid = cid
        self.device = para_dict['device']

        self.features = data_dict['features_list'][cid]     
        self.adj = data_dict['adj']                         
        self.adj_norm = None                                             
        self.train_idx = data_dict['train_idx']             
        self.val_idx = data_dict['val_idx']                 
        self.test_idx = data_dict['test_idx']               
        self.preprocess()                                   

        lr = para_dict['lr']
        weight_decay = para_dict['weight_decay']
        in_dim = self.features.shape[1]
        hid_dim = para_dict['hid_dim']
        out_dim = para_dict['out_dim']

        if para_dict['model'] == 'GCN':
            self.local_model = GCN(nfeat=in_dim, nhid=hid_dim, nemb=out_dim, dropout=0, device=self.device)
        elif para_dict['model'] == 'SGC':
            self.local_model = SGC(nfeat=in_dim, nemb=out_dim, device=self.device)
        self.optimizer = torch.optim.Adam(self.local_model.parameters(), lr=lr, weight_decay=weight_decay)

    def preprocess(self):
        if type(self.adj) is not torch.Tensor:
            self.adj, self.features = utils.to_tensor(self.adj, self.features, device=self.device)
        else:
            self.features = self.features.to(self.device)
            self.adj = self.adj.to(self.device)

        if utils.is_sparse_tensor(self.adj):
            self.adj_norm = utils.normalize_adj_tensor(self.adj, sparse=True)
        else:
            self.adj_norm = utils.normalize_adj_tensor(self.adj)

    def output(self, is_train: bool = False, target_node=None):
        embedding = self.local_model(self.features, self.adj_norm)
        if target_node is None:
            return embedding                            
        else:
            return embedding[[target_node]]             


class Malicious(Client):
    def __init__(self, cid, para_dict, data_dict):
        super(Malicious, self).__init__(cid, para_dict, data_dict)
        self.para_dict = para_dict
        self.data_dict = data_dict

        self.attack_method = para_dict['attack']
        self.adj_ptb = None
        self.adj_ptb_norm = None

    def preprocess_ptb(self):
        if type(self.adj_ptb) is not torch.Tensor:
            if sp.issparse(self.adj_ptb):
                self.adj_ptb = sparse_mx_to_torch_sparse_tensor(self.adj_ptb).to(self.device)
            else:
                self.adj_ptb = torch.FloatTensor(self.adj_ptb).to(self.device)
        else:
            self.adj_ptb = self.adj_ptb.to(self.device)

        if utils.is_sparse_tensor(self.adj_ptb):
            self.adj_ptb_norm = utils.normalize_adj_tensor(self.adj_ptb, sparse=True)
        else:
            self.adj_ptb_norm = utils.normalize_adj_tensor(self.adj_ptb)

    def output(self, is_train: bool = False, target_node=None):
        if is_train:
            embedding = self.local_model(self.features, self.adj_norm)
            return embedding
        else:
            # attack
            if target_node is None:
                raise "attacked node is lacked!"
            if self.attack_method in ['GF', 'GF_pgd', 'Nettack']:
                self.adj_ptb = self.attack(self.attack_method, self.para_dict, self.data_dict, target_node)
                self.preprocess_ptb()
                embedding = self.local_model(self.features, self.adj_ptb_norm)
            elif self.attack_method in ['Gaussian', 'Missing', 'Flipping']:
                embedding = self.attack(self.attack_method, self.para_dict, self.data_dict, target_node)
            else:
                raise "no attack method!"
            if target_node is None:
                return embedding                    
            else:
                return embedding[[target_node]]     

    def attack(self, attack_method, para_dict, data_dict, target_node):
        special = True if attack_method in ['GF', 'GF_pgd', 'Nettack'] else False       
        attack = getattr(importlib.import_module('.'.join(['attack', f'{attack_method}'])), 'attack')
        if special:
            if attack_method in ['GF', 'GF_pgd'] and not hasattr(self, 'shadow_global_model'):
                infer_global_model = getattr(importlib.import_module
                                             ('.'.join(['attack', f'{attack_method}'])), 'infer_global_model')
                self.shadow_global_model = infer_global_model(self, para_dict, data_dict)

            if attack_method == 'Nettack' and not hasattr(self, 'surrogate_model'):
                train_surrogate_model = getattr(importlib.import_module
                                                ('.'.join(['attack', f'{attack_method}'])), 'train_surrogate_model')
                self.surrogate_model = train_surrogate_model(self, para_dict, data_dict)

            modified_adj = attack(self, para_dict, data_dict, target_node)
            return modified_adj
        else:
            embedding = attack(self, para_dict, data_dict, target_node)
            return embedding
