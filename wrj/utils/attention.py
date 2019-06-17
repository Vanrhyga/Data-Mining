import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class attention(nn.Module):
    def __init__(self, embedding_dim, cuda = "cpu"):
        super(attention, self).__init__()

        self.embed_dim = embedding_dim
        self.device = cuda
        # self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim).to(self.device)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim).to(self.device)
        self.att3 = nn.Linear(self.embed_dim, 1).to(self.device)
        self.softmax = nn.Softmax(0)

    def forward(self, feature1, feature2, n_neighs):
        feature2_reps = feature2.repeat(n_neighs, 1)

        x = torch.cat((feature1, feature2_reps), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)

        att = F.softmax(x, dim=0)

        return att
