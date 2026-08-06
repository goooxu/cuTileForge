import torch
import torch.nn as nn

"""
GroupNorm (tier 5, norm)
"""

NUM_GROUPS = 16
NUM_CHANNELS = 4096
BATCH_SIZE = 256
SPATIAL_DIM = 32

class Model(nn.Module):
    """GroupNorm (tier 5, norm)"""

    def __init__(self, num_groups, num_channels):
        super(Model, self).__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels)

    def forward(self, x):
        return self.norm(x)


# Module-level constants for shape configuration
NUM_GROUPS = 16
NUM_CHANNELS = 4096
BATCH_SIZE = 256
SPATIAL_DIM = 32

def get_inputs():
    """Create input tensors suitable for large throughput measurements"""
    # Create a tensor of shape (BATCH_SIZE, NUM_CHANNELS, SPATIAL_DIM, SPATIAL_DIM)
    x = torch.randn(BATCH_SIZE, NUM_CHANNELS, SPATIAL_DIM, SPATIAL_DIM)
    return [x]

def get_init_inputs():
    """Create initialization arguments for the model"""
    return [NUM_GROUPS, NUM_CHANNELS]