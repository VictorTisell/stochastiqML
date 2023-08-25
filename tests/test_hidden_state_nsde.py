import torch
import time
import unittest

from stochastiqML.aa_nsde import HiddenStateNSDE

class TestHiddenStateSDE(unittest.TestCase):

    def setUp(self):
        self.batch_size = 256
        self.t_size = 20
        self.hidden_dim = 32
        self.latent_dim = 16
        self.model = HiddenStateNSDE(hidden_dim=self.hidden_dim,
                                     latent_dim=self.latent_dim, nlayers=2)
        self.model.eval()
        self.x = torch.randn(self.batch_size, self.hidden_dim)
        self.t = torch.linspace(0, 1, steps=self.t_size)

    def test_f_g_methods(self):
        drift = self.model.f(self.t, self.x)
        diffusion = self.model.g(self.t, self.x)
        self.assertEqual(drift.shape, (self.batch_size, self.hidden_dim))
        self.assertEqual(diffusion.shape, (self.batch_size, self.hidden_dim))

    def test_forward_method(self):
        expected_shape = (self.batch_size, self.t_size, self.hidden_dim)
        start = time.time()
        out = self.model(self.t, self.x)
        elapsed_time = time.time() - start
        self.assertEqual(out.shape, expected_shape)
        self.assertTrue(elapsed_time < 1.5)

if __name__ == '__main__':
    unittest.main()
