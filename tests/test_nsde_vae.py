
import unittest
import torch

from stochastiqML.sde_vae import NSDE_VAE, VariationalEncoder
from stochastiqML.sde_vae import Decoder
class TestNSDE_VAE(unittest.TestCase):
    def setUp(self):
        self.input_dim = 64
        self.hidden_dim = 128
        self.latent_dim = 32
        self.nlayers = 2
        self.dropout = 0.2
        self.batch_size = 8
        self.seq_len = 10
        self.nsde_vae = NSDE_VAE(self.input_dim, self.hidden_dim, self.latent_dim, self.nlayers, self.dropout)
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

    def test_initialization(self):
        self.assertIsInstance(self.nsde_vae.encoder, VariationalEncoder)
        self.assertIsInstance(self.nsde_vae.decoder, Decoder)

    def test_forward_pass(self):
        output = self.nsde_vae(self.x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_backward_pass(self):
        output = self.nsde_vae(self.x)
        loss = output.mean()
        loss.backward()

if __name__ == '__main__':
    unittest.main()
