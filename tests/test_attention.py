import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

from stochastiqML.aa_nsde import SelfAttention

class TestSelfAttention(unittest.TestCase):

    def setUp(self):
        self.model = SelfAttention(feature_dim=32)
        self.batch_size = 10
        self.seq_len = 20
        self.feature_dim = 32

    def test_initialization(self):
        self.assertIsInstance(self.model.query, nn.Linear)
        self.assertIsInstance(self.model.key, nn.Linear)
        self.assertIsInstance(self.model.value, nn.Linear)
        self.assertEqual(self.model.query.in_features, self.feature_dim)
        self.assertEqual(self.model.query.out_features, self.feature_dim)

    def test_forward_pass(self):
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        output = self.model(x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.feature_dim))

    def test_attention_weights(self):
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        q = self.model.query(x)
        k = self.model.key(x)
        scores = torch.bmm(q, k.transpose(1,2))/self.feature_dim**0.5
        attention_weights = F.softmax(scores, dim=2)
        self.assertTrue(torch.allclose(attention_weights.sum(dim=2), torch.tensor(1.0)))

    def test_backward_pass(self):
        x = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        output = self.model(x)
        loss = output.mean()
        loss.backward()
        # Check if gradients are computed for the parameters
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient not computed for {name}")

if __name__ == '__main__':
    unittest.main()
