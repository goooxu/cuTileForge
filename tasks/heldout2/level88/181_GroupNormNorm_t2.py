import torch
import torch.nn as nn

# Shape constants for large tensor throughput measurement
BATCH_SIZE = 32
CHANNELS = 256
HEIGHT = 64
WIDTH = 64

class Model(nn.Module):
    """GroupNormNorm (tier 2, norm)"""

    def __init__(self, num_groups=32, num_channels=256, eps=1e-5, affine=True):
        super(Model, self).__init__()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def forward(self, x):
        return self.group_norm(x)

def get_inputs():
    """Create input tensor with shape suitable for throughput measurement"""
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [32, 256, 1e-5, True]