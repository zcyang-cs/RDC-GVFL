from deeprobust.graph.targeted_attack import RND
from deeprobust.graph.data import Pyg2Dpr
import warnings

warnings.filterwarnings('ignore')


def attack(params_dict, data_dict, target_node):
    # =========================================Data======================================================
    # convert Pyg to Dpr
    dataset = data_dict['dataset']
    Dpr_dataset = Pyg2Dpr(dataset)
    Dpr_data = Dpr_dataset
    Dpr_adj, Dpr_labels = Dpr_data.adj, Dpr_data.labels  # adj: scipy.sparse._csr.csr_matrix
    idx_train, idx_val, idx_test = Dpr_data.idx_train, Dpr_data.idx_val, Dpr_data.idx_test
    # =========================================Attack======================================================
    # print("\n==========RND-attack==========")
    # 1. Setup Attack Model
    model = RND()
    # 2. Attack: RND().attack()
    model.attack(Dpr_adj, Dpr_labels, idx_test, target_node, n_perturbations=1)
    modified_adj = model.modified_adj.tocsr()

    return modified_adj
