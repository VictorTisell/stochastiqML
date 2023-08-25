import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, feature_dim:int)->None:
        super().__init__()
        self.feature_dim = feature_dim
        self.query = nn.Linear(feature_dim, feature_dim, bias = False)
        self.key = nn.Linear(feature_dim, feature_dim, bias = False)
        self.value = nn.Linear(feature_dim, feature_dim, bias = False)
    def forward(self, x:torch.tensor)->torch.tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.bmm(q, k.transpose(1,2))/self.feature_dim**0.5
        attention = F.softmax(scores, dim = 2)
        return torch.bmm(attention, v)

