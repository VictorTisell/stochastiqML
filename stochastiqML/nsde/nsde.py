import torch
import torchsde
from typing import Optional
'''
add lipschits continuous drift and diffusion (spectral norm)
'''
class NeuralSDE(torch.nn.Module):
    def __init__(self, input_dim:int,
                 hidden_dim:int,
                 nlayers:int,
                 sde_type:Optional[str] = 'ito',
                 noise_type:Optional[str] = 'diagonal',
                 solver:Optional[str] = 'euler',
                 dropout:Optional[float] = 0.0,
                 lipschitz:Optional[bool] = True)->None:
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.output_activation = None
        if lipschitz:
            regularization = torch.nn.utils.spectral_norm
        self.dropout = torch.nn.Dropout(dropout)
        self.drift_input = torch.nn.Linear(input_dim, hidden_dim)
        self.diffusion_input = torch.nn.Linear(input_dim, hidden_dim)
        self.drift_fc = torch.nn.ModuleList([torch.nn.Sequential(
                                    regularization(torch.nn.Linear(hidden_dim, hidden_dim)) \
                                            if lipschitz else torch.nn.Linear(hidden_dim, hidden_dim) ,
                                    self.dropout,
                                    self.activation)
                                    for _ in range(nlayers)])
        self.diffusion_fc = torch.nn.ModuleList([torch.nn.Sequential(
                                    regularization(torch.nn.Linear(hidden_dim, hidden_dim)) \
                                            if lipschitz else torch.nn.Linear(hidden_dim, hidden_dim),
                                    self.dropout,
                                    self.activation)
                                    for _ in range(nlayers)])
        self.drift_output = torch.nn.Linear(hidden_dim, input_dim)
        self.diffusion_output = torch.nn.Linear(hidden_dim, input_dim)
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
if __name__ == '__main__':
    pass
