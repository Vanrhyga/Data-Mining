import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class componentProcess(nn.Module):
    def __init__(self, feature, feature_size, embedding_dim, cuda = 'cpu'):
        super(componentProcess, self).__init__()

        self.embed_dim = embedding_dim
        self.fea_dim = feature_size
        self.feature = feature
        self.device = cuda
        self.layer = nn.Linear(self.fea_dim, self.embed_dim, bias = False).to(self.device)

    def forward(self):
        component_embed = self.layer(self.feature)

        return component_embed
