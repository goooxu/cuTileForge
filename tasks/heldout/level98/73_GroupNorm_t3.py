import torch
import torch.nn as nn

# Module-level constants for large tensor sizes
BATCH_SIZE = 64
IN_CHANNELS = 256
HEIGHT = 64
WIDTH = 64
GROUPS = 32


class Model(nn.Module):
    """GroupNorm (tier 3, norm)"""

    def __init__(self, num_groups, num_channels):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.norm.eval()  # Make deterministic for benchmarking

    def forward(self, x):
        return self.norm(x)


def get_inputs():
    """Return a list with a single input tensor for benchmarking."""
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    ]


def get_init_inputs():
    """Return arguments to pass to __init__."""
    return [GROUPS, IN_CHANNELS]