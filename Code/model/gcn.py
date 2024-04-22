import torch
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907 \n
    copy from DeepRobust
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(torch.nn.Module):
    """
    2 Layer Partial Graph Convolutional Network(aka. Head GCN), changed from DeepRobust.

    Init:
        model = Head_GCN(nfeat, nhid, nemb, ...)
    Usage:
        model(x, adj_norm) -> embedding
    """

    def __init__(self, nfeat, nhid, nemb, dropout=0.5, lr=0.01, weight_decay=5e-4,
                 with_relu=True, with_bias=True, device=None):
        super(GCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nemb = nemb
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nemb, with_bias=with_bias)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.inner_features = None
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def forward(self, x, adj):
        """
        前向传播生成emb

        :param x: features, type-tensor
        :param adj: normalized adj, type-sparse tensor
        :return: embedding, type-tensor
        """
        if self.with_relu:
            x = self.gc1(x, adj)
            x = F.relu(x)
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        self.output = x
        return x
