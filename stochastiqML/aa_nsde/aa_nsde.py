import torch
import torchsde
from typing import Optional

from stochastiqML.aa_nsde.attention import SelfAttention
from stochastiqML.aa_nsde.initial_state_encoder import InitialStateEncoder

class HiddenStateNSDE(torch.nn.Module):
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

class AttentionAugmentedNSDE(torch.nn.Module):
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
        self.hidden_state = HiddenStateNSDE(hidden_dim = hidden_dim,
                                        latent_dim = latent_dim,
                                        nlayers = nlayers,
                                        dropout = dropout)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        assert initial_state in ['zeros', 'random', 'learned', 'conditional']
        self._initial_state = initial_state
        self._initial_state_encoder = None
        if self._initial_state == 'conditional':
            self._initial_state_encoder = InitialStateEncoder(input_dim,
                                                        latent_dim,
                                                        hidden_dim)
    @property
    def initial_state(self)->str:
        return self._initial_state
    @initial_state.setter
    def initial_state(self, value:str)->None:
        self._initial_state = value
        if self._initial_state == 'conditional' and self._initial_state_encoder is None:
            self._initial_state_encoder = InitialStateEncoder(self.input_dim,
                                                              self.latent_dim,
                                                              self.hidden_dim)
    def get_initial_state(self, x:torch.tensor)->torch.tensor:
        if self._initial_state == 'zeros':
            return torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        elif self._initial_state == 'random':
            raise NotImplementedError("Random initial state not implemented yet.")
        elif self._initial_state == 'learned':
            raise NotImplementedError("Learned initial state not implemented yet.")
        elif self._initial_state == 'conditional':
            mean, logvar =  self._initial_state_encoder(x[:, 0, :])
            return self._initial_state_encoder.reparameterize(mean, logvar)
        else:
            raise ValueError(f"Invalid initial state: {self.initial_state}")
    def forward(self, x:torch.tensor)->torch.tensor:
        x0 = self.get_initial_state(x)
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
    model = AttentionAugmentedNSDE(input_dim = input_dim,
                                hidden_dim=hidden_dim,
                                latent_dim = latent_dim,
                                nlayers = 2,
                                initial_state = 'conditional')
    x = torch.randn(input_shape)
    y = model(x)
