import torch
import torch.nn as nn
import unittest

from stochastiqML.nsde import NeuralSDE

class TestNeuralSDE(unittest.TestCase):

    def setUp(self):
        self.input_dim = 32
        self.hidden_dim = 64
        self.nlayers = 2
        self.batch_size = 10
        self.seq_len = 20

    def test_initialization(self):
        model = NeuralSDE(input_dim=self.input_dim, hidden_dim=self.hidden_dim, nlayers=self.nlayers)
        self.assertIsInstance(model.drift_input, nn.Linear)
        self.assertIsInstance(model.diffusion_input, nn.Linear)
        self.assertEqual(len(model.drift_fc), self.nlayers)
        self.assertEqual(len(model.diffusion_fc), self.nlayers)

    def test_forward_pass(self):
        model = NeuralSDE(input_dim=self.input_dim, hidden_dim=self.hidden_dim, nlayers=self.nlayers)
        x0 = torch.randn(self.batch_size, self.input_dim)
        t = torch.linspace(0, 1, steps=self.seq_len)
        output = model(t, x0)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_backward_pass(self):
        model = NeuralSDE(input_dim=self.input_dim, hidden_dim=self.hidden_dim, nlayers=self.nlayers)
        x0 = torch.randn(self.batch_size, self.input_dim)
        t = torch.linspace(0, 1, steps=self.seq_len)
        output = model(t, x0)
        loss = output.mean()
        loss.backward()
        # Check if gradients are computed for the parameters
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient not computed for {name}")

    def test_lipschitz_regularization(self):
        model = NeuralSDE(input_dim=self.input_dim, hidden_dim=self.hidden_dim, nlayers=self.nlayers, lipschitz=True)
        self.assertIsInstance(model.drift_input, nn.Linear)
        self.assertIsInstance(model.diffusion_input, nn.Linear)
        self.assertEqual(len(model.drift_fc), self.nlayers)
        self.assertEqual(len(model.diffusion_fc), self.nlayers)

if __name__ == '__main__':
    unittest.main()
