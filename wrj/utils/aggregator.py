import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from attention import attention


class aggregator(nn.Module):
    """
    item and user aggregator: for aggregating feature of neighbors (item/user aggregator).
    """
    def __init__(self, u_feature, i_feature, embedding_dim, cuda="cpu",
                    is_user = True):
        super(aggregator, self).__init__()

        self.ufeature = u_feature
        self.ifeature = i_feature
        self.embed_dim = embedding_dim
        self.device = cuda
        self.is_user = is_user
        self.att = attention(self.embed_dim, self.device)

    def forward(self, ui_network):
        embed_matrix = torch.empty(len(ui_network), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(ui_network)):
            interactions = ui_network[i]
            n = len(interactions)

            if self.is_user == True:
                # user component
                neighs_feature = self.ifeature.weight[interactions]
                node_feature = self.ufeature.weight[i]
            else:
                # item component
                neighs_feature = self.ufeature.weight[interactions]
                node_feature = self.ifeature.weight[i]

            att_w = self.att(neighs_feature, node_feature, n)
            embedding = torch.mm(neighs_feature.t(), att_w)
            embedding = embedding.t()

            embed_matrix[i] = embedding

        return embed_matrix
