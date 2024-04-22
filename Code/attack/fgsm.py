import torch
import numpy as np
import torch.nn.functional as F
from utils.utils import aggregate


def fgsm_attack(device, aggregation, num_clients,
                emb_infer, emb_B, surrogate_server, labels_B, target_node, eps=1):
    # cat_emb = torch.cat((clean_embedding_infer_, malicious_local_embedding_), dim=-1)
    cat_emb = aggregate(aggregation, emb_infer, emb_B, num_clients=num_clients)
    cat_emb = cat_emb.cpu().detach()
    cat_emb = cat_emb.to(device)
    base_emb = cat_emb[target_node].clone().cpu().data.numpy()        # [[target_node]] -> [1, 16]
    input = cat_emb[target_node].clone()
    input.requires_grad = True

    surrogate_server.eval()
    pred = surrogate_server(input)
    surrogate_server.zero_grad()
    pred = F.log_softmax(pred, dim=-1)
    loss_target_node = F.nll_loss(pred, labels_B[target_node])
    loss_target_node.backward()
    grad_sign = input.grad.data.cpu().sign().numpy()

    emb_infer_with_noise = base_emb + eps * grad_sign

    input1 = torch.FloatTensor(emb_infer_with_noise).to(device)
    pred_pert = surrogate_server(input1)
    pred_pert = F.log_softmax(pred_pert, dim=-1)
    k_i = np.argmax(pred_pert.cpu().data.numpy())
    if_success = 0
    label_target = labels_B[target_node]
    if label_target != k_i:
        if_success += 1

    return emb_infer_with_noise, if_success
