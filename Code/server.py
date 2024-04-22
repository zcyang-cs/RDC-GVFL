import copy
import numpy as np
import scipy
import torch
from tqdm import tqdm

from model import FCN
import torch.nn.functional as F
from utils.utils import accuracy
from flow import logger


class Server(object):
    def __init__(self, para_dict, data_dict):
        # 架构
        self.aggregation = para_dict['aggregation']

        # client列表
        self.clients = []
        self.device = para_dict['device']

        # server拥有的数据
        self.labels = data_dict['labels']  # ndarray
        self.train_idx = data_dict['train_idx']  # ndarray
        self.val_idx = data_dict['val_idx']  # ndarray
        self.test_idx = data_dict['test_idx']  # ndarray
        self.num_classes = self.labels.max() + 1
        # move to gpu
        self.preprocess()  # labels: tensor(gpu); other var: ndarray

        # server 训练相关
        if self.aggregation in ['sum', 'avg']:
            self.global_in_dim = para_dict['out_dim']
        elif self.aggregation == 'concat':
            self.global_in_dim = para_dict['out_dim'] * para_dict['num_clients']
        self.global_out_dim = self.num_classes
        self.global_model = FCN(self.global_in_dim, self.global_out_dim)

        self.lr = para_dict['lr']
        self.weight_decay = para_dict['weight_decay']
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)  # 优化器
        self.epochs = para_dict['epochs']  # 训练轮数

        # server 防御相关
        self.real_malicious = para_dict['malicious']  # 真实的malicious,用于防御部分的历史emb收集(主要是因为那部分代码没有统一)
        # self.lr_ae = para_dict['lr_ae']

    def preprocess(self):
        """
        将labels转成tensor,移到gpu上
        :return:
        """
        self.labels = torch.LongTensor(self.labels)
        if torch.cuda.is_available():
            self.labels = self.labels.to(self.device)

    def append(self, client):
        self.clients.append(client)

    def aggregate(self, current_emb, previous_emb):
        aggregation = self.aggregation
        if aggregation in ['sum', 'avg']:
            next_emb = current_emb + previous_emb
        elif aggregation == 'concat':
            next_emb = torch.cat((current_emb, previous_emb), dim=-1)
        else:
            raise 'there is no aggregation method!'
        return next_emb

    def communicate(self, is_train=False, target_node=None):
        """

        :param is_train:
        :param target_node:
        :return:
        """
        aggregation = self.aggregation
        for idx, client in enumerate(self.clients):
            if is_train:
                client.local_model.train()
                client.optimizer.zero_grad()
                local_emb = client.output(is_train=True, target_node=target_node)
            else:
                client.local_model.eval()
                local_emb = client.output(is_train=False, target_node=target_node)

            if idx == 0:
                global_emb = local_emb
            else:
                global_emb = self.aggregate(local_emb, global_emb)
        if aggregation == 'avg':
            global_emb /= len(self.clients)
        return global_emb

    def train(self):
        """
        训练: 所有节点一起
        :return:
        """
        # local model, global model移入GPU
        self.global_model.to(self.device)
        for client in self.clients:
            client.local_model.to(self.device)

        # 训练
        local_emb_list = []
        for epoch in range(self.epochs):
            self.global_model.train()
            self.optimizer.zero_grad()

            # 1. client产生local embedding, 所有client的输出求和得到global embedding
            global_emb = self.communicate(is_train=True)

            # 2. server接收global embedding, server返回预测
            pred = self.global_model(global_emb)  # pred: (2708, 7)
            pred = F.log_softmax(pred, dim=-1)  # 概率归一化后的pred: (2708, 7)

            # 3. 在server端进行反向传播
            loss = F.nll_loss(pred[self.train_idx], self.labels[self.train_idx])
            loss.backward()

            self.optimizer.step()
            for client in self.clients:
                client.optimizer.step()
            if epoch % 50 == 0:
                logger.info("FL training...Epoch: %d, train loss: %f" % (epoch, loss.cpu().detach().numpy()))

            # 保存所有client在最后一轮输出的embedding，因为后续会用到
            if epoch == self.epochs - 1:
                pred_Server = pred
                for idx, client in enumerate(self.clients):
                    local_emb = client.output(is_train=True)
                    local_emb_list.append(local_emb)

        torch.save(pred_Server, 'saved_models/pred_Server.pth')
        torch.save(local_emb_list, 'saved_models/client_embedding_list.pth')
        torch.save(self.global_model, 'saved_models/global_model.pkl')

    def test(self):
        """
        测试: 对每个节点单独判断是否分类对,然后对结果求和
        :return:
        """
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
            global_emb = self.communicate(is_train=False, target_node=target_node)

            # server接收global embedding, 返回预测
            pred = self.global_model(global_emb)

            # 计算准确率acc
            outputS = F.log_softmax(pred, dim=-1)
            # outputS[[0]]==outputS: (1, 7)         labels: (2708)
            correct += accuracy(outputS[[0]], self.labels[[target_node]]).item()
        acc = int(correct) / int(len(self.test_idx))

        logger.info(f"FL testing...server's test acc:\t{acc}\t")
        # logger.info("==========real server's training and testing is done==========")

    def single_predict(self, target_node):
        self.global_model.eval()
        global_emb = self.communicate(is_train=False, target_node=target_node)  # global_emb: (1, 16)
        pred = self.global_model(global_emb)  # pred: (1, 7)
        outputS = F.log_softmax(pred, dim=-1)
        # outputS.max(1)[1]: (1)即一维张量, outputS.max(1)[1]: ()即一个数
        return outputS.max(1)[1].item()









