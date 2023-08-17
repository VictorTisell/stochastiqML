import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, feature_dim:int):
        super().__init__()
        self.feature_dim = feature_dim
        self.attn = nn.Linear(feature_dim*2, feature_dim,
                              bias = False)
        self.v = nn.Parameter(torch.rand(feature_dim))
    def forward(self, hidden_states:torch.tensor)->torch.tensor:
        seq_len = hidden_states.size(1)
        h_i = hidden_states.unsqueeze(2).repeat(1,1,seq_len,1)
        h_j = hidden_states.unsqueeze(1).repeat(1,seq_len,1,1)
        energy = torch.tanh(self.attn(torch.cat([h_i, h_j], dim = 3)))
        energy = energy.permute(0,3,1,2)
        v = self.v.repeat(hidden_states.size(0),1).unsqueeze(1)
        attention = torch.matmul(v, energy).squeeze(1)
        attention_weights = F.softmax(attention, dim=1)
        return torch.bmm(attention_weights, hidden_states)

