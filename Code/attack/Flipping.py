


def attack(malicious_client, para_dict, data_dict, target_node):
    scale = para_dict['scale']
    device = para_dict['device']
    embedding = malicious_client.local_model(malicious_client.features, malicious_client.adj_norm)
    embedding = embedding * -scale

    return embedding.to(device)
