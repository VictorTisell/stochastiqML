import torch
from typing import Optional

from aa_nsde import HiddenStateNSDE

class PerturbNet(torch.nn.Module):
    def __init__(self, hidden_dim:int):
        super().__init__()
        self.intensities = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.fc = torch.nn.Linear(hidden_dim, hidden_dim)
    def forward(self, hidden_states:torch.tensor)->torch.tensor:
        pass
