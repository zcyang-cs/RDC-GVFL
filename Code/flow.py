import argparse
import importlib
import torch
import random
import numpy as np
from utils.allocateGPU import get_available_gpu, allocate_gpu
from utils.logger.basic_logger import Logger

logger = None


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_paras():
    parser = argparse.ArgumentParser(description='GVFL robustness arguments')
    parser.add_argument('--seed', type=int, default=30, help='Random seed.')
    parser.add_argument('--device', type=int, default=None, help='device.')
    parser.add_argument('--dataset', type=str, default='Cora', choices=['Cora', 'Cora_ML', 'Citeseer', 'Pubmed'])
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--model', type=str, default='GCN', help='Local model', choices=['GCN', 'SGC'])
    parser.add_argument('--aggregation', type=str, default='sum', help='GVFL aggregation method',
                        choices=['sum', 'avg', 'concat'])

    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epoches')
    parser.add_argument('--hid_dim', type=list, default=32, help='Dimensions of hidden layers / local hid dim')
    parser.add_argument('--out_dim', type=list, default=16, help='Dimensions of out layers / local emb dim')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep  probability)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight for l2 loss on embedding matrix')

    parser.add_argument('--attack', type=str, default='GF', help='Attack algo',
                        choices=['Nettack', 'GF', 'GF_pgd', 'RND', 'Gaussian', 'Missing', 'Flipping'])
    parser.add_argument('--rand_malicious', default=False, help='Whether to randomly choose malicious',
                        action='store_true')                        # 输入为True,不输入默认为False
    parser.add_argument('--malicious', type=int, default=None, help='Malicious client index')
    parser.add_argument('--scale', type=float, default=1., help='scale of Flipping attack')

    parser.add_argument('--detection', type=str, default=None, help='Detection algo', choices=['shapley', 'distance'])
    parser.add_argument('--defense', type=str, default=None, help='Defense algo',
                        choices=['sim', 'drop', 'weight', 'krum', 'dp', 'top-k', 'copur'])
    # parser.add_argument('--lr_ae', type=float, default=0.01, help='Initial AutoEncoder learning rate')

    parser.add_argument('--store_log', help='bool controls whether to store log and default value is False',
                        action="store_true", default=False)         # 对日志进行控制,不输入默认为False
    args = parser.parse_args()

    set_seed(args.seed)
    if type(args.device) == int:
        args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    else:
        gpu_to_use, gpu_available = get_available_gpu()
        allocate_gpu()
        args.device = torch.device(f"cuda:{gpu_to_use}" if torch.cuda.is_available() else "cpu")
    if args.rand_malicious:
        num_malicious_client = 1
        malicious_client = random.sample(range(0, args.num_clients), num_malicious_client)
        args.malicious = malicious_client[0]
    para_dict = vars(args)

    return para_dict


def initialize(para_dict, data_dict):
    # log_name = \                          # log不记录aggregation了,因为都是基于sum在做的
    #     f"{para_dict['dataset']}_s{para_dict['seed']}_cnum{para_dict['num_clients']}_{para_dict['model']}" \
    #     if para_dict['malicious'] is None \
    #     else f"{para_dict['dataset']}_s{para_dict['seed']}_cnum{para_dict['num_clients']}_{para_dict['model']}" \
    #          f"_m{para_dict['malicious']}_A|{para_dict['attack']}|_d|{para_dict['detection']}|_D|{para_dict['defense']}|"
    log_name = \
        f"{para_dict['aggregation']}_cnum{para_dict['num_clients']}_{para_dict['model']}_D|{para_dict['defense']}|" \
        if para_dict['malicious'] is None \
        else f"{para_dict['aggregation']}_cnum{para_dict['num_clients']}_{para_dict['model']}" \
             f"_m{para_dict['malicious']}_A|{para_dict['attack']}|_d|{para_dict['detection']}|_D|{para_dict['defense']}|"

    global logger
    logger = Logger(name=log_name, para_dict=para_dict)
    logger.info(f"device: {para_dict['device']}")
    logger.info(f"malicious client: {para_dict['malicious']}")

    # 动态导入Server和Client,这样方便和logger一起使用
    server = getattr(importlib.import_module('server'), 'Server')(para_dict, data_dict)
    for client_index in range(para_dict['num_clients']):
        if client_index == para_dict['malicious']:
            client = getattr(importlib.import_module('client'), 'Malicious')(cid=client_index,
                                                                             para_dict=para_dict, data_dict=data_dict)
        else:
            client = getattr(importlib.import_module('client'), 'Client')(cid=client_index,
                                                                          para_dict=para_dict, data_dict=data_dict)
        server.append(client)
    return server

