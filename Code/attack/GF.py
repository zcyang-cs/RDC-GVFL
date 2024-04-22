from attack.inverse import inverse_attack
from attack.fgsm import fgsm_attack
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from copy import deepcopy
from deeprobust.graph.data import Dpr2Pyg
from flow import logger
from utils.utils import aggregate
import warnings

warnings.filterwarnings('ignore')


# 注意：和GVFL-Multi存在精度差异，主要是因为client_embedding_list有微小差异，GRN的初始loss也都差不多，但在训练过程中差异被不断放大
def infer_global_model(malicious_client, params_dict, data_dict):
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

    logger.info("==========infer global model==========")
    num_clients = params_dict['num_clients']
    aggregation = params_dict['aggregation']
    malicious_client_index = malicious_client.cid
    epochs, lr, weight_decay = params_dict['epochs'], params_dict['lr'], params_dict['weight_decay']
    local_embedding_dim = params_dict['out_dim']
    device = params_dict['device']

    dataset = data_dict['dataset']
    features, labels = data_dict['features'], data_dict['labels']
    test_idx = data_dict['test_idx']

    # 数据处理、移入gpu
    if params_dict['dataset'] != 'Cora_ML':         # todo:修改
        data = Dpr2Pyg(dataset)[0]
    labels = torch.from_numpy(deepcopy(labels)).to(device)
    test_idx = torch.from_numpy(deepcopy(test_idx)).to(device)

    # 加载训练阶段最后一轮的emb和服务器预测
    client_embedding_list = torch.load('saved_models/client_embedding_list.pth')
    pred_Server = torch.load('saved_models/pred_Server.pth')

    # =========================================Attack======================================================
    # **********************STEP1: 对抗训练推测global embedding**********************
    malicious_local_embedding = client_embedding_list[malicious_client_index].detach()  # 已经在GPU上了
    pred_server = pred_Server.detach()

    # GRN generator
    if aggregation in ['sum', 'avg']:
        G = nn.Sequential(
            nn.Linear(local_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, local_embedding_dim)
        )
        G.to(device)  # G移入GPU
        # Server discriminator
        D = torch.load('saved_models/global_model.pkl')
        D.to(device)  # D移入GPU
        # 生成noise;移入GPU
        noise = torch.rand([features.shape[0], local_embedding_dim]).to(device)

    elif aggregation == 'concat':
        G = nn.Sequential(
            nn.Linear(num_clients * local_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, (num_clients - 1) * local_embedding_dim)
        )
        G.to(device)  # G移入GPU
        # Server discriminator
        D = torch.load('saved_models/global_model.pkl')
        D.to(device)  # D移入GPU
        # 生成noise;移入GPU
        noise = torch.rand([features.shape[0], (num_clients - 1) * local_embedding_dim]).to(device)
    else:
        raise 'no aggregation!'

    # 训练GRN
    optimizer_GRN = torch.optim.Adam(G.parameters(), lr=lr, weight_decay=weight_decay)
    D.eval()
    # malicious_global_embedding = noise + malicious_local_embedding
    malicious_global_embedding = aggregate(aggregation, noise, malicious_local_embedding, num_clients=num_clients)

    epochs_steal = 2000
    # epochs_steal = 2500
    for epoch in range(epochs_steal):
        G.train()

        optimizer_GRN.zero_grad()
        output_embedding = G(malicious_global_embedding)

        output_embedding = aggregate(aggregation, output_embedding, malicious_local_embedding, num_clients=num_clients)
        pred = D(output_embedding)
        pred = F.log_softmax(pred, dim=-1)  # pred: malicious_global_embedding得到的预测
        loss = F.mse_loss(pred, pred_server)  # pred_server: clean_global_embedding得到的预测
        loss.backward()
        optimizer_GRN.step()
        if epoch % 200 == 0:
            logger.info("GRN training...Epoch: %d, train loss: %f" % (epoch, loss.cpu().detach().numpy()))
        if epoch == epochs_steal - 1:  # 第1999轮保存推断结果
            # pred_Server_infer: 拿malicious_global_embedding得到的预测 (2708, 7)
            global pred_Server_infer
            pred_Server_infer = pred

            # clean_embedding_infer：推测出的其他正常用户的embedding (2708, 16 * (cnum - 1))
            global clean_embedding_infer
            if aggregation == 'sum':
                clean_embedding_infer = output_embedding - malicious_local_embedding
            elif aggregation == 'concat':
                clean_embedding_infer = output_embedding.split(noise.shape[1], dim=1)[0]
            elif aggregation == 'avg':
                clean_embedding_infer = output_embedding * num_clients - malicious_local_embedding
            else:
                raise 'no aggregation!'

    # **********************STEP2: train shadow server**********************
    malicious_local_embedding_ = malicious_local_embedding.detach()  # 恶意用户自己的embedding(已知)
    pred_Server_ = pred_Server.detach()  # 真实预测(已知)
    clean_embedding_infer_ = clean_embedding_infer.detach()  # 推断出的正常用户的embedding(推测)

    # shadow server
    if aggregation in ['sum', 'avg']:
        surrogate_server = nn.Sequential(
            nn.Linear(local_embedding_dim, int(labels.max() + 1)),
        )
    elif aggregation == 'concat':
        surrogate_server = nn.Sequential(
            nn.Linear(local_embedding_dim * num_clients, int(labels.max() + 1)),
        )
    else:
        raise 'no aggregation!'
    surrogate_server.to(device)  # 将shadow server移入GPU

    optimizer_surrogate_server = torch.optim.Adam(surrogate_server.parameters(), lr=0.01, weight_decay=weight_decay)

    # cat_embedding = clean_embedding_infer_ + malicious_local_embedding_
    cat_embedding = aggregate(aggregation, clean_embedding_infer_, malicious_local_embedding_, num_clients=num_clients)

    epochs = 200
    # epochs = 500
    for epoch in range(epochs):
        optimizer_surrogate_server.zero_grad()
        pred = surrogate_server(cat_embedding)
        pred = F.log_softmax(pred, dim=-1)
        # 计算代理server的预测pred和真实server的预测pred_Server之间的MSE Loss
        loss = F.mse_loss(pred, pred_Server_)
        loss.backward()
        optimizer_surrogate_server.step()
        if epoch % 100 == 0:
            logger.info("shadow server training...Epoch: %d, train loss: %f" % (epoch, loss.cpu().detach().numpy()))

    # 评估shadow server和真实server的性能
    surrogate_server.eval()
    with torch.no_grad():
        pred = surrogate_server(cat_embedding)
        pred = pred.argmax(dim=1)
        correct = (pred[test_idx] == labels[test_idx]).sum()
        # acc = int(correct) / int(data.test_mask.sum())
        acc = int(correct) / int(len(labels[test_idx]))         # todo: 同上
        logger.info(f"shadow server testing...shadow server's acc:\t{acc}\t")
    logger.info("==========shadow server's training and testing is done==========")

    return surrogate_server


# 和GVFL-Detector精度差异，理由同上
def attack(malicious_client, params_dict, data_dict, target_node):
    aggregation = params_dict['aggregation']

    target_node = [target_node]
    target_node = np.array(target_node)

    features, adj, labels = data_dict['features'], data_dict['adj'], data_dict['labels']
    adj_ = adj

    num_clients = params_dict['num_clients']
    device = params_dict['device']

    pred_Server = torch.load('saved_models/pred_Server.pth')
    client_embedding_list = torch.load('saved_models/client_embedding_list.pth')

    malicious_client_index = malicious_client.cid
    malicious_local_embedding = client_embedding_list[malicious_client_index].detach()  # 已经在GPU上了

    surrogate_server = malicious_client.shadow_global_model

    # **********************数据处理、数据移入GPU**********************
    labels = torch.from_numpy(deepcopy(labels)).to(device)

    # =========================================Attack======================================================
    # logger.info("\n==========GF-attack==========")

    # **********************STEP3: fgsm attack + adversarial edge inference**********************
    # 预测出的标签
    pred_labels = pred_Server.max(dim=1)[1].type_as(labels)
    # logger.info(pred_labels)

    shadow_server = surrogate_server.to(device)  # shadow server
    malicious_local_embedding_ = malicious_local_embedding.detach()  # 恶意用户自己的embedding(已知)
    pred_Server_ = pred_Server.detach()  # 真实预测向量(已知)
    clean_embedding_infer_ = clean_embedding_infer.detach()  # 推断出的正常用户的embedding(推测)

    i = 0

    # fgsm attack
    # emb_infer_with_noise: (1, 48)
    # emb_infer_with_noise, if_succ = fgsm_attack(device, clean_embedding_infer_, malicious_local_embedding_,
    #                                             shadow_server, pred_labels, target_node, eps=0.01)
    emb_infer_with_noise, if_succ = fgsm_attack(device, aggregation, num_clients,
                                                clean_embedding_infer_, malicious_local_embedding_,
                                                shadow_server, pred_labels, target_node, eps=0.01)
    i += if_succ

    # adversarial edge inference
    malicious_local_model = malicious_client.local_model
    malicious_features = malicious_client.features
    # A -> \hat A
    malicious_adj = adj_.copy()
    malicious_adj_copy = adj_.copy()

    # 确定扰动数量
    degrees = data_dict['adj'].sum(0).A1
    n_perturbation = int(degrees[target_node])

    adversarial_edge = inverse_attack(device, num_clients, malicious_local_model, aggregation,
                                      emb_infer_with_noise, clean_embedding_infer_,
                                      target_node, malicious_adj_copy, malicious_features)
    # edge_index_B = edge_index
    # 结点对(1708, 1042)之间加了一条边
    # adversarial_edge = inverse_attack(device, num_clients, malicious_local_model, emb_infer_with_noise,
    #                                   target_node, malicious_adj_copy, malicious_features)
    # logger.info(adversarial_edge)             # 打印出扰动的边
    modified_adj = malicious_adj.copy().tolil()

    for edge in adversarial_edge:
        edge = edge.transpose()
        if modified_adj[edge[0], edge[1]] != 0:
            # 删除边
            modified_adj[edge[0], edge[1]] = 0
            modified_adj[edge[1], edge[0]] = 0
        else:
            # 增加边
            modified_adj[edge[0], edge[1]] = 1
            modified_adj[edge[1], edge[0]] = 1

    # A -> GCN中的\hat A
    modified_adj = modified_adj.tocsr()  # coo -> csr

    return modified_adj
