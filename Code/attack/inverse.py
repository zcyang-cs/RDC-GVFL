import copy

import torch
import numpy as np
import torch.nn.functional as F


def normalize_adj_tensor(adj, sparse, device):
    mx = adj + torch.eye(adj.shape[0]).to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-1/2).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv @ mx
    mx = mx @ r_mat_inv
    return mx


def inverse_attack(device, client_num, model_B, aggregation,
                   emb_infer_with_noise, clean_emb_infer, target_node, A_B, X_B, n_perturbation=1):
    t = target_node[0]
    clean_emb_infer = copy.deepcopy(clean_emb_infer[target_node].cpu().numpy())

    if aggregation == 'sum':
        emb_B_noise = emb_infer_with_noise - clean_emb_infer  
    elif aggregation == 'avg':
        emb_B_noise = emb_infer_with_noise * client_num - clean_emb_infer
    elif aggregation == 'concat':
        emb_B_noise = emb_infer_with_noise[:, (client_num - 1) * 16:client_num * 16]
    else:
        raise 'no aggregation!'
    emb_B_noise = torch.FloatTensor(emb_B_noise).to(device)

    model_B.eval()
    A_B = torch.FloatTensor(A_B.todense()).to(device)   
    A_B.requires_grad = True

    adj_B = normalize_adj_tensor(A_B, False, device)
    emb = model_B(X_B, adj_B)   
    model_B.zero_grad()

    loss = -F.mse_loss(emb[target_node], emb_B_noise) + 1   

    grad = torch.autograd.grad(loss, A_B)[0].data.cpu().numpy()
    grad = (grad[target_node] + grad[:, target_node])

    adversarial_edge = []
    for i in range(n_perturbation):
        idx = grad[t].argmax()
        adversarial_edge.append(np.array([t, idx]))
        np.delete(grad[t], idx)
    return adversarial_edge
