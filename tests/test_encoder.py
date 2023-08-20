import torch
import unittest

from stochastiqML.sde_vae import Encoder

class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.input_size = 32
        self.hidden_size = 64
        self.latent_size = 16
        self.encoder = Encoder(self.input_size, self.hidden_size, self.latent_size)
        self.x = torch.randn(10, 20, self.input_size)  # Batch of 10, sequence length of 20

    def test_forward(self):
        out = self.encoder(self.x)
        self.assertEqual(out.shape, (10, 20, self.latent_size))

    def test_backward(self):
        out = self.encoder(self.x)
        out.mean().backward()
if __name__ == '__main__':
    unittest.main()
