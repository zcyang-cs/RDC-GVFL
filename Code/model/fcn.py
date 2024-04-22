import torch
import torch.nn as nn


# server的全连接网络
class FCN(torch.nn.Module):
    def __init__(self, global_embedding_dim, num_classes):
        super(FCN, self).__init__()
        self.classifier = nn.Sequential(
            # todo: previous
            # nn.ReLU(),                    # dont be used
            nn.Linear(global_embedding_dim, num_classes)
            # todo: add relu
            # nn.Linear(global_embedding_dim, 16),
            # nn.ReLU(),
            # nn.Linear(16, num_classes)
        )

    def forward(self, global_embedding):
        # 输出概率分布
        out = self.classifier(global_embedding)
        return out
