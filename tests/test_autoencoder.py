import unittest
import torch

from stochastiqML.sde_vae import Encoder, Decoder

class TestAutoEncoder(unittest.TestCase):
    def setUp(self):
        self.input_size = 32
        self.hidden_size = 64
        self.latent_size = 16
        self.output_size = 32
        self.encoder = Encoder(self.input_size, self.hidden_size, self.latent_size)
        self.decoder = Decoder(self.latent_size, self.hidden_size, self.output_size)
        self.x = torch.randn(10, 20, self.input_size)  # Batch of 10, sequence length of 20

    def test_forward(self):
        encoded = self.encoder(self.x)
        self.assertEqual(encoded.shape, (10, 20, self.latent_size))
        
        decoded = self.decoder(encoded)
        self.assertEqual(decoded.shape, (10, 20, self.output_size))

    def test_backward(self):
        encoded = self.encoder(self.x)
        decoded = self.decoder(encoded)
        decoded.mean().backward()

if __name__ == '__main__':
    unittest.main()

