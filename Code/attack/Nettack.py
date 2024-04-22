from copy import deepcopy
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.data import Pyg2Dpr
from deeprobust.graph.defense import GCN as Full_GCN
import warnings
import scipy.sparse as sp
from flow import logger

warnings.filterwarnings('ignore')


def train_surrogate_model(malicious_client, para_dict, data_dict):
    malicious_features = deepcopy(malicious_client.features.cpu()).numpy()
    malicious_features = sp.csr_matrix(malicious_features)

    adj, labels = data_dict['adj'], data_dict['labels']
    idx_train, idx_val, idx_test = data_dict['train_idx'], data_dict['val_idx'], data_dict['test_idx']

    logger.info("==========train surrogate model==========")
    num_class = labels.max().item() + 1
    device = para_dict['device']
    lr, weight_decay = para_dict['lr'], para_dict['weight_decay']
    local_embedding_dim = para_dict['out_dim']

    surrogate = Full_GCN(nfeat=malicious_features.shape[1], nclass=num_class,
                         nhid=local_embedding_dim, lr=lr, weight_decay=weight_decay,
                         dropout=0, with_relu=False, with_bias=False, device=device)
    surrogate = surrogate.to(device)

    surrogate.fit(malicious_features, adj, labels, idx_train, idx_val, patience=30)

    return surrogate


def attack(malicious_client, para_dict, data_dict, target_node):
    malicious_features = deepcopy(malicious_client.features.cpu()).numpy()
    malicious_features = sp.csr_matrix(malicious_features)

    surrogate = malicious_client.surrogate_model
    device = para_dict['device']

    adj, labels = data_dict['adj'], data_dict['labels']
    degrees = data_dict['adj'].sum(0).A1
    n_perturbations = int(degrees[target_node])     

    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False,
                    device=device)
    model = model.to(device)
    model.attack(malicious_features, adj, labels, target_node, n_perturbations, verbose=False)
    modified_adj = model.modified_adj       

    modified_adj = modified_adj.tocsr()     

    return modified_adj
