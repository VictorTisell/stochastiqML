import torch
import torchsde
from typing import Optional

from stochastiqML.sde_rnn.attention import SelfAttention

class HiddenState(torch.nn.Module):
    def __init__(self, hidden_dim:int,
                 latent_dim:int,
                 nlayers:int,
                 sde_type:Optional[str] = 'ito',
                 noise_type:Optional[str] = 'diagonal',
                 solver:Optional[str] = 'euler',
                 dropout:Optional[float] = 0.0)->None:
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.output_activation = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(dropout)
        self.drift_input = torch.nn.Linear(hidden_dim, latent_dim)
        self.diffusion_input = torch.nn.Linear(hidden_dim, latent_dim)
        self.drift_fc = torch.nn.ModuleList([torch.nn.Sequential(
                                    torch.nn.Linear(latent_dim, latent_dim),
                                    self.dropout,
                                    self.activation)
                                    for _ in range(nlayers)])
        self.diffusion_fc = torch.nn.ModuleList([torch.nn.Sequential(
                                    torch.nn.Linear(latent_dim, latent_dim),
                                    self.dropout,
                                    self.activation)
                                    for _ in range(nlayers)])
        self.drift_output = torch.nn.Linear(latent_dim, hidden_dim)
        self.diffusion_output = torch.nn.Linear(latent_dim, hidden_dim)
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.solver = solver
    def f(self, t:torch.tensor, x:torch.tensor)->torch.tensor:
        x = self.drift_input(x)
        for fc in self.drift_fc:
            x = fc(x)
        x = self.drift_output(x)
        return self.output_activation(x) if self.output_activation else x
    def g(self, t:torch.tensor, x:torch.tensor)->torch.tensor:
        x = self.diffusion_input(x)
        for fc in self.diffusion_fc:
            x = fc(x)
        x  = self.diffusion_output(x)
        return self.output_activation(x) if self.output_activation else x
    def forward(self, t:torch.tensor, x0:torch.tensor)->torch.tensor:
        return torchsde.sdeint(self, x0, t, method = self.solver).swapaxes(0, 1)
'''
add learnable initial state
'''
class AttentionAugmentedSDE(torch.nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int,
                 latent_dim:int, nlayers:int,
                 initial_state:Optional[str] = 'zeros',
                 dropout:Optional[float] = 0.2)->None:
        super().__init__()
        self.attention = SelfAttention(feature_dim = input_dim)
        self.activation = torch.nn.ReLU()
        self.output_activation = None
        self.dropout = torch.nn.Dropout(dropout)
        self.fc_input = torch.nn.Linear(input_dim + hidden_dim, latent_dim)
        self.fc_block = torch.nn.ModuleList([torch.nn.Sequential(
                                    torch.nn.Linear(latent_dim, latent_dim),
                                    self.dropout,
                                    self.activation)
                                    for _ in range(nlayers)])
        self.fc_output = torch.nn.Linear(latent_dim, input_dim)
        self.hidden_state = HiddenState(hidden_dim = hidden_dim,
                                        latent_dim = latent_dim,
                                        nlayers = nlayers,
                                        dropout = dropout)
        self.initial_state = initial_state
        self.hidden_dim = hidden_dim
        assert initial_state in ['zeros', 'random', 'learned']
    def forward(self, x:torch.tensor)->torch.tensor:
        if self.initial_state == 'zeros':
            x0 = torch.zeros(x.size(0), self.hidden_dim)
        elif self.initial_state == 'random':
            raise NotImplementedError
        elif self.initial_state == 'learned':
            raise NotImplementedError
        context = self.attention(x)
        hidden_state = self.hidden_state(x0 = x0, 
                                         t = torch.linspace(0, 1, steps = x.size(1)))
        x = torch.cat([context, hidden_state], dim = 2)
        x = self.fc_input(x)
        for fc in self.fc_block:
            x = fc(x)
        x = self.fc_output(x)
        return self.output_activation(x) if self.output_activation else x

if __name__ == '__main__':
    batch_size = 100
    t_size = 20
    hidden_dim = 4
    input_dim = 10
    latent_dim = 16
    input_shape = (batch_size, t_size, input_dim)
    t = torch.linspace(0, 1, steps = t_size)
    model = AttentionAugmentedSDE(input_dim = input_dim,
                                hidden_dim=hidden_dim,
                                latent_dim = latent_dim,
                                nlayers = 2)
    x = torch.randn(input_shape)
    y = model(x)
    print(y)
