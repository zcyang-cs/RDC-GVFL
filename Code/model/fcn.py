import torch
import torch.nn as nn


class FCN(torch.nn.Module):
    def __init__(self, global_embedding_dim, num_classes):
        super(FCN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(global_embedding_dim, num_classes)
        )

    def forward(self, global_embedding):
        out = self.classifier(global_embedding)
        return out
