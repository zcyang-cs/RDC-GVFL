import torch


def attack(malicious_client, para_dict, data_dict, target_node):
    node_num = malicious_client.features.shape[0]
    out_dim = para_dict['out_dim']
    device = para_dict['device']
    embedding = malicious_client.local_model(malicious_client.features, malicious_client.adj_norm)
    gaussian_noise = torch.randn(node_num, out_dim).to(device)
    embedding += gaussian_noise

    return embedding.to(device)
