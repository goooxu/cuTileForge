import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNorm (tier 3, norm)"""
    
    def __init__(self, num_features=64):
        super(Model, self).__init__()
        self.num_features = num_features
        # Initialize weight with ones for RMS normalization
        self.weight = nn.Parameter(torch.ones(num_features))
    
    def forward(self, x):
        # Compute RMS normalization
        rms = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-5)
        normalized = x / rms
        return normalized * self.weight.view(1, -1, 1, 1)

# Module-level constants for tensor shapes
BATCH_SIZE = 3
IN_CHANNELS = 64
HEIGHT = 12
WIDTH = 12
def get_inputs():
    """Returns a list of input tensors for the forward pass."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns a list of arguments for the __init__ method."""
    return [IN_CHANNELS]