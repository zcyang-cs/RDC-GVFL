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


def infer_global_model(malicious_client, params_dict, data_dict):
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

    if params_dict['dataset'] != 'Cora_ML':        
        data = Dpr2Pyg(dataset)[0]
    labels = torch.from_numpy(deepcopy(labels)).to(device)
    test_idx = torch.from_numpy(deepcopy(test_idx)).to(device)

    client_embedding_list = torch.load('saved_models/client_embedding_list.pth')
    pred_Server = torch.load('saved_models/pred_Server.pth')

    malicious_local_embedding = client_embedding_list[malicious_client_index].detach()  
    pred_server = pred_Server.detach()

    if aggregation in ['sum', 'avg']:
        G = nn.Sequential(
            nn.Linear(local_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, local_embedding_dim)
        )
        G.to(device)  
        D = torch.load('saved_models/global_model.pkl')
        D.to(device) 
        noise = torch.rand([features.shape[0], local_embedding_dim]).to(device)

    elif aggregation == 'concat':
        G = nn.Sequential(
            nn.Linear(num_clients * local_embedding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, (num_clients - 1) * local_embedding_dim)
        )
        G.to(device)  
        D = torch.load('saved_models/global_model.pkl')
        D.to(device)  
        noise = torch.rand([features.shape[0], (num_clients - 1) * local_embedding_dim]).to(device)
    else:
        raise 'no aggregation!'

    optimizer_GRN = torch.optim.Adam(G.parameters(), lr=lr, weight_decay=weight_decay)
    D.eval()
    malicious_global_embedding = aggregate(aggregation, noise, malicious_local_embedding, num_clients=num_clients)

    epochs_steal = 2000
    for epoch in range(epochs_steal):
        G.train()

        optimizer_GRN.zero_grad()
        output_embedding = G(malicious_global_embedding)

        output_embedding = aggregate(aggregation, output_embedding, malicious_local_embedding, num_clients=num_clients)
        pred = D(output_embedding)
        pred = F.log_softmax(pred, dim=-1)  
        loss = F.mse_loss(pred, pred_server)  
        loss.backward()
        optimizer_GRN.step()
        if epoch % 200 == 0:
            logger.info("GRN training...Epoch: %d, train loss: %f" % (epoch, loss.cpu().detach().numpy()))
        if epoch == epochs_steal - 1:  
            global pred_Server_infer
            pred_Server_infer = pred

            global clean_embedding_infer
            if aggregation == 'sum':
                clean_embedding_infer = output_embedding - malicious_local_embedding
            elif aggregation == 'concat':
                clean_embedding_infer = output_embedding.split(noise.shape[1], dim=1)[0]
            elif aggregation == 'avg':
                clean_embedding_infer = output_embedding * num_clients - malicious_local_embedding
            else:
                raise 'no aggregation!'

    malicious_local_embedding_ = malicious_local_embedding.detach()  
    pred_Server_ = pred_Server.detach()  
    clean_embedding_infer_ = clean_embedding_infer.detach()  

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
    surrogate_server.to(device)  

    optimizer_surrogate_server = torch.optim.Adam(surrogate_server.parameters(), lr=0.01, weight_decay=weight_decay)

    cat_embedding = aggregate(aggregation, clean_embedding_infer_, malicious_local_embedding_, num_clients=num_clients)

    epochs = 200
    for epoch in range(epochs):
        optimizer_surrogate_server.zero_grad()
        pred = surrogate_server(cat_embedding)
        pred = F.log_softmax(pred, dim=-1)
        loss = F.mse_loss(pred, pred_Server_)
        loss.backward()
        optimizer_surrogate_server.step()
        if epoch % 100 == 0:
            logger.info("shadow server training...Epoch: %d, train loss: %f" % (epoch, loss.cpu().detach().numpy()))

    surrogate_server.eval()
    with torch.no_grad():
        pred = surrogate_server(cat_embedding)
        pred = pred.argmax(dim=1)
        correct = (pred[test_idx] == labels[test_idx]).sum()
        acc = int(correct) / int(len(labels[test_idx]))         
        logger.info(f"shadow server testing...shadow server's acc:\t{acc}\t")
    logger.info("==========shadow server's training and testing is done==========")

    return surrogate_server


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
    malicious_local_embedding = client_embedding_list[malicious_client_index].detach()  

    surrogate_server = malicious_client.shadow_global_model

    labels = torch.from_numpy(deepcopy(labels)).to(device)
    pred_labels = pred_Server.max(dim=1)[1].type_as(labels)

    shadow_server = surrogate_server.to(device)  
    malicious_local_embedding_ = malicious_local_embedding.detach()  
    pred_Server_ = pred_Server.detach()  
    clean_embedding_infer_ = clean_embedding_infer.detach()  

    i = 0
    emb_infer_with_noise, if_succ = fgsm_attack(device, aggregation, num_clients,
                                                clean_embedding_infer_, malicious_local_embedding_,
                                                shadow_server, pred_labels, target_node, eps=0.01)
    i += if_succ

    malicious_local_model = malicious_client.local_model
    malicious_features = malicious_client.features
    malicious_adj = adj_.copy()
    malicious_adj_copy = adj_.copy()

    degrees = data_dict['adj'].sum(0).A1
    n_perturbation = int(degrees[target_node])

    adversarial_edge = inverse_attack(device, num_clients, malicious_local_model, aggregation,
                                      emb_infer_with_noise, clean_embedding_infer_,
                                      target_node, malicious_adj_copy, malicious_features)
    modified_adj = malicious_adj.copy().tolil()

    for edge in adversarial_edge:
        edge = edge.transpose()
        if modified_adj[edge[0], edge[1]] != 0:
            modified_adj[edge[0], edge[1]] = 0
            modified_adj[edge[1], edge[0]] = 0
        else:
            modified_adj[edge[0], edge[1]] = 1
            modified_adj[edge[1], edge[0]] = 1

    modified_adj = modified_adj.tocsr() 

    return modified_adj
