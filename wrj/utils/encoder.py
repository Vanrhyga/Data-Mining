import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


class encoder(nn.Module):
    def __init__(self, features, embedding_dim, adj, ratings, aggregator,
                    cuda="cpu", is_user = True):
        super(encoder, self).__init__()

        self.features = features
        self.adj = adj
        self.ratings = ratings
        self.aggregator = aggregator
        self.embed_dim = embedding_dim
        self.device = cuda
        self.layer = nn.Linear(2 * self.embed_dim, self.embed_dim).to(self.device)

    def forward(self, nodes):
        interactions = []
        _ratings = []
        for node in nodes:
            interactions.append(self.adj[int(node)])
            _ratings.append(self.ratings[int(node)])

        neigh_features = self.aggregator.forward(nodes, interactions, _ratings)  # user-item network
        self_features = self.features.weight[nodes]
        # self-connection could be considered.

        combined = torch.cat([self_features, neigh_features], dim=1)
        component_embed_matrix = F.relu(self.layer(combined))

        return component_embed_matrix
