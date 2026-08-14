import torch
import torch.nn as nn

class Model(nn.Module):
    """GroupNorm (tier 2, norm)"""
    
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)
    
    def forward(self, x):
        return self.group_norm(x)


# Module-level constants for shape configuration
NUM_GROUPS = 2
NUM_CHANNELS = 4
BATCH_SIZE = 1
HEIGHT = 2
WIDTH = 2

def get_inputs():
    """Returns a list of tensors to pass to forward."""
    # Create input tensor with shape (batch, channels, height, width)
    x = torch.randn(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return [NUM_GROUPS, NUM_CHANNELS]