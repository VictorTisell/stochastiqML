import unittest
import torch

from stochastiqML.sde_vae import Decoder

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.latent_size = 16
        self.hidden_size = 64
        self.output_size = 32
        self.decoder = Decoder(self.latent_size, self.hidden_size, self.output_size)
        self.x = torch.randn(10, 20, self.latent_size)  # Batch of 10, sequence length of 20

    def test_forward(self):
        out = self.decoder(self.x)
        self.assertEqual(out.shape, (10, 20, self.output_size))

    def test_backward(self):
        out = self.decoder(self.x)
        out.mean().backward()
if __name__ == '__main__':
    unittest.main()
