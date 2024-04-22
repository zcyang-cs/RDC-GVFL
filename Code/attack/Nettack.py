from copy import deepcopy
from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.data import Pyg2Dpr
from deeprobust.graph.defense import GCN as Full_GCN
import warnings
import scipy.sparse as sp
from flow import logger

warnings.filterwarnings('ignore')


def train_surrogate_model(malicious_client, para_dict, data_dict):
    """ +++++++++++++++++++++++++++++data_dict格式+++++++++++++++++++++++++++++
        'dataset': Dpr_dataset,                     # Dataset(deeprobust)
        'data': Dpr_dataset,                        # Dataset(deeprobust)
        'features': features,                       # numpy.ndarray
        'labels': labels,                           # numpy.ndarray
        'adj': adj,                                 # scipy.sparse._csr.csr_matrix
        'train_idx': train_idx,                     # numpy.ndarray
        'val_idx': val_idx,                         # numpy.ndarray
        'test_idx': test_idx                        # numpy.ndarray
        'features_list': partitioned features       # list, list[i]:numpy.ndarray
    """
    # 为了和DeepRobust数据格式兼容，因此转成csr_matrix(=====很关键=====)
    malicious_features = deepcopy(malicious_client.features.cpu()).numpy()
    malicious_features = sp.csr_matrix(malicious_features)

    adj, labels = data_dict['adj'], data_dict['labels']
    idx_train, idx_val, idx_test = data_dict['train_idx'], data_dict['val_idx'], data_dict['test_idx']

    logger.info("==========train surrogate model==========")
    # =====================================设置代理模型，训练代理模型===========================================
    num_class = labels.max().item() + 1
    device = para_dict['device']
    lr, weight_decay = para_dict['lr'], para_dict['weight_decay']
    local_embedding_dim = para_dict['out_dim']

    surrogate = Full_GCN(nfeat=malicious_features.shape[1], nclass=num_class,
                         nhid=local_embedding_dim, lr=lr, weight_decay=weight_decay,
                         dropout=0, with_relu=False, with_bias=False, device=device)
    surrogate = surrogate.to(device)

    # surrogate.fit会自动将变量移入GPU,自动将features和adj转成sparse_tensor,以及对adj做normalize
    surrogate.fit(malicious_features, adj, labels, idx_train, idx_val, patience=30)
    # surrogate.fit(malicious_features, adj, labels, idx_train, None, patience=30)

    return surrogate


def attack(malicious_client, para_dict, data_dict, target_node):
    # 攻击者相关data
    malicious_features = deepcopy(malicious_client.features.cpu()).numpy()
    malicious_features = sp.csr_matrix(malicious_features)

    surrogate = malicious_client.surrogate_model
    device = para_dict['device']

    adj, labels = data_dict['adj'], data_dict['labels']
    # =========================================Attack======================================================
    degrees = data_dict['adj'].sum(0).A1
    n_perturbations = int(degrees[target_node])     # 扰动数量与目标节点的度相关

    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=False,
                    device=device)
    model = model.to(device)
    model.attack(malicious_features, adj, labels, target_node, n_perturbations, verbose=False)
    # logger.info(model.structure_perturbations)    # 打印出扰动的边
    modified_adj = model.modified_adj       # modified_adj: scipy.sparse._lil.lil_matrix

    modified_adj = modified_adj.tocsr()     # modified_adj: scipy.sparse._lil.lil_matrix -> csr_matrix

    return modified_adj

    # 直接返回攻击后的embedding
    # malicious_client.adj_ptb = malicious_client.attack(malicious_client.attack_method, malicious_client.para_dict,
    #                                                    malicious_client.data_dict, target_node)
    # malicious_client.preprocess_ptb()
    # embedding = malicious_client.local_model(malicious_client.features, malicious_client.adj_ptb_norm)
    # return embedding
