import torch
import numpy as np
import torch.nn.functional as F
from utils.utils import aggregate


def pgd_attack(device, aggregation, num_clients,
               emb_infer, emb_B, surrogate_server, labels_B, target_node, alpha=2/255, eps=0.5):
    # 得到input
    cat_emb = aggregate(aggregation, emb_infer, emb_B, num_clients=num_clients)
    cat_emb = cat_emb.cpu().detach()
    cat_emb = cat_emb.to(device)
    # base_emb = cat_emb[target_node].clone().cpu().data.numpy()        # [[target_node]] -> [1, 16]
    base_emb = cat_emb[target_node].clone()
    input = cat_emb[target_node].clone()

    # pgd_attack
    iters = 50
    for i in range(iters):
        input.requires_grad = True
        # 得到标签
        surrogate_server.eval()
        pred = surrogate_server(input)
        surrogate_server.zero_grad()
        pred = F.log_softmax(pred, dim=-1)
        # 对x求导反向传播
        loss_target_node = F.nll_loss(pred, labels_B[target_node])
        loss_target_node.backward()
        # fgsm
        # grad_sign = input.grad.data.cpu().sign().numpy()
        grad_sign = input.grad.data.sign()
        emb_infer_with_noise = base_emb + alpha * grad_sign

        # 投影
        eta = torch.clamp(emb_infer_with_noise - input, min=-eps, max=eps)
        # 进行下一轮对抗样本的生成。破坏之前的计算图
        input = torch.clamp(input + eta, min=0, max=1).detach_()

        # input = torch.from_numpy(emb_infer_with_noise).cpu().detach()
        # input = input.to(device).clone()

    emb_infer_with_noise = input.clone().cpu().data.numpy()
    if_success = 1
    # input1 = torch.FloatTensor(input).to(device)
    # pred_pert = surrogate_server(input1)
    # pred_pert = F.log_softmax(pred_pert, dim=-1)
    # k_i = np.argmax(pred_pert.cpu().data.numpy())
    # if_success = 0
    # label_target = labels_B[target_node]
    # if label_target != k_i:
    #     if_success += 1

    return emb_infer_with_noise, if_success
