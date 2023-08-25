import torch
import unittest

from stochastiqML.sde_vae import VariationalEncoder

""" class TestVariationalEncoder(unittest.TestCase): """
"""     def test_initialization(self): """
"""         input_dim = 64 """
"""         latent_dim = 32 """
"""         nheads = 4 """
"""         nlayers = 2 """
"""         dim_feedforward = 128 """
"""         dropout = 0.2 """
""""""
"""         encoder = VariationalEncoder(input_dim, latent_dim, nheads, nlayers, dim_feedforward, dropout) """
""""""
"""         self.assertIsInstance(encoder.encoder, torch.nn.TransformerEncoder) """
"""         self.assertIsInstance(encoder.relu, torch.nn.ReLU) """
"""         self.assertIsInstance(encoder.mean_fc, torch.nn.Linear) """
"""         self.assertIsInstance(encoder.logvar_fc, torch.nn.Linear) """
""""""
"""     def test_forward_pass(self): """
"""         input_dim = 64 """
"""         latent_dim = 32 """
"""         nheads = 4 """
"""         nlayers = 2 """
"""         dim_feedforward = 128 """
"""         dropout = 0.2 """
"""         batch_size = 8 """
"""         seq_len = 10 """
""""""
"""         encoder = VariationalEncoder(input_dim, latent_dim, nheads, nlayers, dim_feedforward, dropout) """
"""         x = torch.rand(batch_size, seq_len, input_dim) """
"""         mu, logvar = encoder(x) """
""""""
"""         self.assertEqual(mu.shape, (batch_size, seq_len, latent_dim)) """
"""         self.assertEqual(logvar.shape, (batch_size, seq_len, latent_dim)) """
""""""
"""     def test_backward_pass(self): """
"""         input_dim = 64 """
"""         latent_dim = 32 """
"""         nheads = 4 """
"""         nlayers = 2 """
"""         dim_feedforward = 128 """
"""         dropout = 0.2 """
"""         batch_size = 8 """
"""         seq_len = 10 """
""""""
"""         encoder = VariationalEncoder(input_dim, latent_dim, nheads, nlayers, dim_feedforward, dropout) """
"""         x = torch.rand(batch_size, seq_len, input_dim) """
"""         mu, logvar = encoder(x) """
""""""
"""         loss = torch.mean(mu + logvar) """
"""         loss.backward() """
""""""
"""         # Check if gradients are computed """
"""         for param in encoder.parameters(): """
"""             self.assertIsNotNone(param.grad) """
""""""
class TestVariationalEncoder(unittest.TestCase):
    def setUp(self):
        self.input_dim = 64
        self.latent_dim = 32
        self.nheads = 4
        self.nlayers = 2
        self.dim_feedforward = 128
        self.dropout = 0.2
        self.batch_size = 8
        self.seq_len = 10
        self.encoder = VariationalEncoder(self.input_dim, self.latent_dim, self.nheads, self.nlayers, self.dim_feedforward, self.dropout)
        self.x = torch.randn(self.batch_size, self.seq_len, self.input_dim)

    def test_forward(self):
        mu, logvar = self.encoder(self.x)
        self.assertEqual(mu.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_backward(self):
        mu, logvar = self.encoder(self.x)
        loss = (mu + logvar).mean()
        loss.backward()
if __name__ == '__main__':
    unittest.main()
