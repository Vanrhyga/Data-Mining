import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class decomposer(nn.Module):
    def __init__(self, feature, feature_size, embedding_dim, cuda = 'cpu'):
        super(decomposer, self).__init__()

        self.feature = feature
        self.fea_dim = feature_size
        self.embed_dim = embedding_dim
        self.device = cuda
        self.layer = nn.Linear(self.fea_dim, self.embed_dim, bias = False)

    def forward(self):
        component_embed = self.layer(self.feature).to(self.device)

        return component_embed
