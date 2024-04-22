
# ++++++++++++++++++++++++++++++++++++defense++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++直接移除++++++++++++++++++
def test_wo_malicious(self, malicious):
    # local model, global model移入GPU
    self.global_model.to(self.device)
    for client in self.clients:
        client.local_model.to(self.device)
    self.global_model.eval()

    target_nodes = self.test_idx
    correct = 0
    for j in tqdm(target_nodes):
        target_node = j.item()
        # client产生embedding, 连接起来构成global embedding
        global_emb = self.communicate_wo_malicious(target_node, malicious)

        # server接收global embedding, 返回预测
        pred = self.global_model(global_emb)

        # 计算准确率acc
        outputS = F.log_softmax(pred, dim=-1)
        # outputS[[0]]==outputS: (1, 7)         labels: (2708)
        correct += accuracy(outputS[[0]], self.labels[[target_node]]).item()
    acc = int(correct) / int(len(self.test_idx))

    logger.info(f"FL defense...server's test acc:\t{acc}\t")


def communicate_wo_malicious(self, target_node, malicious):
    aggregation = self.aggregation  # sum/avg
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