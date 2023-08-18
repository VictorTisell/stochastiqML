import torch
import torch.nn as nn
import unittest
from stochastiqML.sde_rnn import AttentionAugmentedSDE, SelfAttention

class TestAttentionAugmentedSDE(unittest.TestCase):

    def setUp(self):
        self.model = AttentionAugmentedSDE(input_dim=32, hidden_dim=64, latent_dim=128, nlayers=2)
        self.batch_size = 10
        self.seq_len = 20
        self.input_dim = 32

    def test_initialization(self):
        self.assertIsInstance(self.model.attention, SelfAttention)
        self.assertIsInstance(self.model.fc_input, nn.Linear)
        self.assertIsInstance(self.model.fc_output, nn.Linear)
        self.assertEqual(len(self.model.fc_block), 2)

    def test_forward_pass(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_hidden_state_initialization_zeros(self):
        self.model.initial_state = 'zeros'
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.input_dim))

    def test_hidden_state_initialization_random(self):
        self.model.initial_state = 'random'
        with self.assertRaises(NotImplementedError):
            x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
            self.model(x)

    def test_hidden_state_initialization_learned(self):
        self.model.initial_state = 'learned'
        with self.assertRaises(NotImplementedError):
            x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
            self.model(x)

    def test_backward_pass(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.model(x)
        loss = output.mean()
        loss.backward()
        # Check if gradients are computed for the parameters
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient not computed for {name}")

if __name__ == '__main__':
    unittest.main()
