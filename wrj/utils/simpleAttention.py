import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class simpleAttention(nn.Module):
    def __init__(self, embedding1, embedding2, embedding3, embedding_dim, cuda = 'cpu'):
        super(simpleAttention, self).__init__()

        self.embed_dim = embedding_dim
        self.device = cuda
        self.att1 = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)
        self.softmax = nn.Softmax()

    def forward(self):
        x = torch.cat((embedding1, embedding2, embedding3), 1)
        x = F.relu(self.att1(x).to(self.device))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x).to(self.device))
        x = F.dropout(x, training=self.training)
        x = self.att3(x).to(self.device)

        att_w = F.softmax(x, dim=0)

        embedding_list = torch.cat((embedding1, embedding2, embedding3), 2)
        final_embed_matrix = torch.mm(embedding_list.t(), att_w)
        final_embed_matrix = final_embed_matrix.t()

        return final_embed_matrix
