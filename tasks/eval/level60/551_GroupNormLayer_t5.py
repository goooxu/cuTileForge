import torch
import torch.nn as nn

class Model(nn.Module):
    """GroupNormLayer (tier 5, norm)"""

    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        self.group_norm.eval()  # Ensure deterministic behavior

    def forward(self, x):
        return self.group_norm(x)


# Module-level constants for shapes
NUM_GROUPS = 2
NUM_CHANNELS = 4
BATCH_SIZE = 2
HEIGHT = 3
WIDTH = 3

def get_inputs():
    """Return input tensor for forward pass"""
    return [torch.randn(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for __init__"""
    return [NUM_GROUPS, NUM_CHANNELS]