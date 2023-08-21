import unittest
import torch

from stochastiqML.aa_nsde import InitialStateEncoder

class TestInitialStateEncoder(unittest.TestCase):

    def setUp(self):
        # Set up a sample tensor and the model for testing
        self.input_size = 32
        self.hidden_size = 64
        self.latent_size = 16
        self.x = torch.randn(10, self.input_size)  # Batch of 10 with input_size of 32
        self.model = InitialStateEncoder(self.input_size, self.hidden_size, self.latent_size)

    def test_forward_pass(self):
        # Test the forward method
        mean, logvar = self.model(self.x)
        self.assertEqual(mean.shape, (10, self.latent_size))
        self.assertEqual(logvar.shape, (10, self.latent_size))

    def test_backward_pass(self):
        # Test the backward pass
        mean, logvar = self.model(self.x)
        z = self.model.reparameterize(mean, logvar)
        loss = z.mean()
        loss.backward()
        # Check if gradients are populated
        self.assertIsNotNone(self.model.fc_mean.weight.grad)
        self.assertIsNotNone(self.model.fc_logvar.weight.grad)

if __name__ == '__main__':
    unittest.main()
