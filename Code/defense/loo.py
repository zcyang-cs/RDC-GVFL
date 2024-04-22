def test_wo_malicious(self, malicious):
    self.global_model.to(self.device)
    for client in self.clients:
        client.local_model.to(self.device)
    self.global_model.eval()

    target_nodes = self.test_idx
    correct = 0
    for j in tqdm(target_nodes):
        target_node = j.item()
        global_emb = self.communicate_wo_malicious(target_node, malicious)

        pred = self.global_model(global_emb)

        outputS = F.log_softmax(pred, dim=-1)
        correct += accuracy(outputS[[0]], self.labels[[target_node]]).item()
    acc = int(correct) / int(len(self.test_idx))

    logger.info(f"FL defense...server's test acc:\t{acc}\t")


def communicate_wo_malicious(self, target_node, malicious):
    aggregation = self.aggregation  
    clients = [client for idx, client in enumerate(self.clients) if idx != malicious]
    for idx, client in enumerate(clients):
        client.local_model.eval()
        local_emb = client.output(is_train=False, target_node=target_node)

        if idx == 0:
            global_emb = local_emb
        else:
            global_emb = self.aggregate(local_emb, global_emb)
    if aggregation == 'avg':
        global_emb /= len(self.clients)
    return global_emb
