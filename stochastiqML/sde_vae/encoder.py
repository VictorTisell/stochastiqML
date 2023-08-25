from typing import Optional
import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_size:int,
                 hidden_size:int,
                 latent_size:int,
                 num_layers:Optional[int] = 10,
                 num_heads:Optional[int] = 8,
                 dropout:Optional[float] = 0.1,
                 dim_feedforward:Optional[int] = 2048):
        super().__init__()
        self.embedding = torch.nn.Linear(input_size, hidden_size)
        if not dim_feedforward:
            dim_feedforward = hidden_size
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden_size,
                                                        nhead = num_heads,
                                                        dim_feedforward = dim_feedforward,
                                                        dropout = dropout)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(hidden_size, latent_size)
        self.output_activation = torch.nn.ReLU()
    def forward(self, x:torch.tensor)->torch.tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.relu(x)
        x = self.fc(x)
        return self.output_activation(x)
