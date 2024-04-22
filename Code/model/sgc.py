import math
import torch
from torch.nn.parameter import Parameter


class GraphConvolution(torch.nn.Module):
    """
    Simple SGC layer, similar to https://arxiv.org/abs/1609.02907 \n
    based on GraphConv copy from DeepRobust
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


class SGC(torch.nn.Module):
    """
    相比GCN, 只有一层GCN, 没有dropout, 没有relu
    1 Layer SGC Network(aka. Head GCN), changed from GCN.

    Init:
        model = Head_GCN(nfeat, nemb, ...)
    Usage:
        model(x, adj_norm) -> embedding
    """

    def __init__(self, nfeat, nemb, lr=0.01, weight_decay=5e-4, with_bias=True, device=None):
        super(SGC, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.nemb = nemb
        self.gc1 = GraphConvolution(nfeat, nemb, with_bias=with_bias)
        self.lr = lr
        self.weight_decay = weight_decay

        self.with_bias = with_bias
        self.inner_features = None
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None

    def initialize(self):
        self.gc1.reset_parameters()

    def forward(self, x, adj):
        """
        前向传播生成emb

        :param x: features, type-tensor
        :param adj: normalized adj, type-sparse tensor
        :return: embedding, type-tensor
        """
        x = self.gc1(x, adj)
        self.output = x
        return x
