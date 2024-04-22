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
        self.aggregation = para_dict['aggregation']
        self.clients = []
        self.device = para_dict['device']
        self.labels = data_dict['labels']  # ndarray
        self.train_idx = data_dict['train_idx']  # ndarray
        self.val_idx = data_dict['val_idx']  # ndarray
        self.test_idx = data_dict['test_idx']  # ndarray
        self.num_classes = self.labels.max() + 1
        self.preprocess()  
        if self.aggregation in ['sum', 'avg']:
            self.global_in_dim = para_dict['out_dim']
        elif self.aggregation == 'concat':
            self.global_in_dim = para_dict['out_dim'] * para_dict['num_clients']
        self.global_out_dim = self.num_classes
        self.global_model = FCN(self.global_in_dim, self.global_out_dim)
        self.lr = para_dict['lr']
        self.weight_decay = para_dict['weight_decay']
        self.optimizer = torch.optim.Adam(self.global_model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)  
        self.epochs = para_dict['epochs']  
        self.real_malicious = para_dict['malicious']  


    def preprocess(self):
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
        self.global_model.to(self.device)
        for client in self.clients:
            client.local_model.to(self.device)

        local_emb_list = []
        for epoch in range(self.epochs):
            self.global_model.train()
            self.optimizer.zero_grad()
            global_emb = self.communicate(is_train=True)
            pred = self.global_model(global_emb)  
            pred = F.log_softmax(pred, dim=-1)  
            loss = F.nll_loss(pred[self.train_idx], self.labels[self.train_idx])
            loss.backward()

            self.optimizer.step()
            for client in self.clients:
                client.optimizer.step()
            if epoch % 50 == 0:
                logger.info("FL training...Epoch: %d, train loss: %f" % (epoch, loss.cpu().detach().numpy()))

            if epoch == self.epochs - 1:
                pred_Server = pred
                for idx, client in enumerate(self.clients):
                    local_emb = client.output(is_train=True)
                    local_emb_list.append(local_emb)

        torch.save(pred_Server, 'saved_models/pred_Server.pth')
        torch.save(local_emb_list, 'saved_models/client_embedding_list.pth')
        torch.save(self.global_model, 'saved_models/global_model.pkl')

    def test(self):
        self.global_model.to(self.device)
        for client in self.clients:
            client.local_model.to(self.device)
        self.global_model.eval()

        target_nodes = self.test_idx
        correct = 0
        for j in tqdm(target_nodes):
            target_node = j.item()
            global_emb = self.communicate(is_train=False, target_node=target_node)
            pred = self.global_model(global_emb)
            outputS = F.log_softmax(pred, dim=-1)
            correct += accuracy(outputS[[0]], self.labels[[target_node]]).item()
        acc = int(correct) / int(len(self.test_idx))

        logger.info(f"FL testing...server's test acc:\t{acc}\t")

    def single_predict(self, target_node):
        self.global_model.eval()
        global_emb = self.communicate(is_train=False, target_node=target_node)  
        pred = self.global_model(global_emb) 
        outputS = F.log_softmax(pred, dim=-1)
        return outputS.max(1)[1].item()
