import torch


def attack(malicious_client, para_dict, data_dict, target_node):
    node_num = malicious_client.features.shape[0]
    out_dim = para_dict['out_dim']
    device = para_dict['device']
    embedding = torch.zeros(node_num, out_dim)

    return embedding.to(device)
