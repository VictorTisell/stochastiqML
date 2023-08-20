import torch
from typing import Optional

class Decoder(torch.nn.Module):
    def __init__(self, latent_size:int,
                 hidden_size:int,
                 output_size:int,
                 num_layers:Optional[int] = 10,
                 num_heads:Optional[int] = 8,
                 dropout:Optional[float] = 0.1,
                 dim_feedforward:Optional[int] = 2048):
        super().__init__()
        self.l1 = torch.nn.Linear(latent_size, hidden_size)
        if not dim_feedforward:
            dim_feedforward = hidden_size
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model = hidden_size,
                                                         nhead = num_heads,
                                                         dim_feedforward = dim_feedforward,
                                                         dropout=dropout)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers= num_layers)
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.output_activation = None
    def forward(self, x:torch.tensor)->torch.tensor:
        x = self.l1(x)
        x = self.relu(x)
        x = self.decoder(x, x)
        x = self.relu(x)
        x = self.fc(x)
        x = self.output_activation(x) if self.output_activation else x
        return x

if __name__ == '__main__':
    pass
