import torch
import time

from stochastiqML.aa_nsde import HiddenStateNSDE

def test_SDE():
    # Initialize the SDE model
    batch_size = 256
    t_size = 20
    hidden_dim = 32
    latent_dim = 16
    expected_shape = (batch_size, t_size, hidden_dim)
    model = HiddenStateNSDE(hidden_dim=hidden_dim,
                latent_dim=latent_dim, nlayers=2)
    
    # Ensure model is in evaluation mode
    model.eval()
    # Create dummy input tensor
    x = torch.randn(batch_size, hidden_dim)  # Batch of 10 with hidden_dim of 32
    t = torch.linspace(0, 1, steps = t_size)  # Time tensor
    # Test the f and g methods
    drift = model.f(t, x)
    diffusion = model.g(t, x)
    assert drift.shape == (batch_size, hidden_dim), f"Expected shape ({batch_size}, {hidden_dim}), but got {drift.shape}"
    assert diffusion.shape == (batch_size, hidden_dim), f"Expected shape ({batch_size}, {hidden_dim}), but got {diffusion.shape}"
    
    # Test the forward method
    start = time.time()
    out = model(t, x)
    elapsed_time = time.time()-start
    
    assert out.shape == expected_shape, f"Expected shape ({expected_shape}), but got {out.shape}"
    assert elapsed_time < 1.5, f"Forward pass took too long: {elapsed_time}"
    print(f"Forward pass completed in {elapsed_time:.2f} seconds")
    print("All tests passed!")
if __name__ == '__main__':
    test_SDE()
