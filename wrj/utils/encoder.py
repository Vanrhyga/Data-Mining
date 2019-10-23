import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, feature, embedding_dim, adj, aggregator,
                    cuda="cpu"):
        super(encoder, self).__init__()

        self.feature = feature
        self.embed_dim = embedding_dim
        self.adj = adj
        self.aggregator = aggregator
        self.device = cuda
        self.layer = nn.Linear(self.embed_dim * 2, self.embed_dim)

    def forward(self):
        neigh_feature = self.aggregator.forward(adj).to(self.device)  # user-item network

        # self-connection could be considered.
        combined = torch.cat([self.feature, neigh_feature], dim = 1)
        component_embed_matrix = F.relu(self.layer(combined))

        return component_embed_matrix
