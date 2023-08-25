import torch
import torchsde
from typing import Optional
from .encoder import Encoder
from .decoder import Decoder
from ..nsde import NeuralSDE


'''
will do project refactor when everything is working

recall to fix consistent variable names
'''
class VariationalEncoder(torch.nn.Module):
    def __init__(self, input_dim:int,
                 latent_dim:int,
                 nheads:int,
                 nlayers:int,
                 dim_feedforward:int,
                 dropout:Optional[float] = 0.2):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model = input_dim,
                                                         nhead = nheads,
                                                         dim_feedforward = dim_feedforward,
                                                         dropout = dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers = nlayers)
        self.relu = torch.nn.ReLU()
        self.mean_fc = torch.nn.Linear(input_dim, latent_dim)
        self.logvar_fc = torch.nn.Linear(input_dim, latent_dim)

    def forward(self, x:torch.tensor)->torch.tensor:
        x = self.encoder(x)
        x = self.relu(x)
        x = x.mean(dim = 1)
        mu = self.mean_fc(x)
        logvar = self.logvar_fc(x)
        return mu, logvar
    def reparametrize(self, mu:torch.tensor, logvar:torch.tensor)->torch.tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
class NSDE_VAE(torch.nn.Module):
    def __init__(self,input_dim:int,
                 hidden_dim:int,
                 latent_dim:int,
                 nlayers:int,
                 dropout:Optional[float] = 0.2,
                 sde_type:Optional[str] = 'ito',
                 noise_type:Optional[str] = 'diagonal',
                 solver:Optional['str'] = 'euler',
                 lipschitz:Optional[bool] = False):
        super().__init__()
        self.encoder = VariationalEncoder(input_dim = input_dim,
                               latent_dim = latent_dim,
                               nlayers = nlayers,
                               dropout = dropout,
                               nheads = 8,
                               dim_feedforward = 32)
        self.decoder = Decoder(latent_size = latent_dim,
                               hidden_size = hidden_dim,
                               output_size = input_dim,
                               num_layers = nlayers,
                               dropout = dropout,
                               num_heads = 8,
                               dim_feedforward = 32)
        self.latent_nsde = NeuralSDE(input_dim = latent_dim,
                                hidden_dim = hidden_dim,
                                nlayers = nlayers,
                                sde_type = sde_type,
                                noise_type = noise_type,
                                dropout = dropout,
                                lipschitz = lipschitz)
    def forward(self, x:torch.tensor)->torch.tensor:
        mu, logvar = self.encoder(x)
        z0 = self.encoder.reparametrize(mu, logvar)
        t = torch.linspace(0, 1, x.size(1))
        z = self.latent_nsde(t,z0)
        return self.decoder(z)
