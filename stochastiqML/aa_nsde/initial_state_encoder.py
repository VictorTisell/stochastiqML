import torch
from typing import Optional

class InitialStateEncoder(torch.nn.Module):
    def __init__(self, input_size:int,
                 hidden_size:int,
                 latent_size:int,
                 nlayers:Optional[int] = 10,
                 dropout:Optional[float] = 0.1):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc_1 = torch.nn.Linear(input_size, hidden_size)
        self.fc_block = torch.nn.ModuleList([torch.nn.Sequential(
                                    torch.nn.Linear(hidden_size, hidden_size),
                                    self.dropout,
                                    self.relu)
                                    for _ in range(nlayers)])
        self.fc_mean = torch.nn.Linear(hidden_size, latent_size)
        self.fc_logvar = torch.nn.Linear(hidden_size, latent_size)
    def forward(self, x:torch.tensor)->torch.tensor:
        x = self.fc_1(x)
        for fc in self.fc_block:
            x = fc(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar
    def reparameterize(self, mean:torch.tensor, logvar:torch.tensor)->torch.tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + eps*std
