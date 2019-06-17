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
    def __init__(self, u_feature, i_feature, embedding_dim,
                    cuda="cpu", is_user = True):
        super(aggregator, self).__init__()

        self.is_user = is_user
        self.u_feature = u_feature
        self.i_feature = i_feature
        self.device = cuda
        self.embed_dim = embedding_dim
        self.att = attention(self.embed_dim, self.device)

    def forward(self, nodes, ui_network, ratings):
        embed_matrix = torch.empty(len(ui_network), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(ui_network)):
            interaction = ui_network[i]
            n_items = len(interaction)
            label = ratings[i]

            if self.is_user == True:
                # user component
                neighs_feature = self.i_feature.weight[interaction]
                node = self.u_feature.weight[nodes[i]]
            else:
                # item component
                neighs_feature = self.u_feature.weight[interaction]
                node = self.i_feature.weight[nodes[i]]

            att_w = self.att(neighs_feature, node, n_items)
            embedding = torch.mm(neighs_feature.t(), att_w)
            embedding = embedding.t()

            embed_matrix[i] = embedding

        return embed_matrix
