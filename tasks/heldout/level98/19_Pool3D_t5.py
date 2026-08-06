import torch
import torch.nn as nn

class Model(nn.Module):
    """Pool3D (tier 5, pool)"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super(Model, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.pool.eval()

    def forward(self, x):
        return self.pool(x)

# Module-level constants for shape configuration
N, C, D, H, W = 2, 3, 64, 64, 64  # batch_size, channels, depth, height, width
KERNEL_SIZE = (2, 2, 2)
STRIDE = (2, 2, 2)
PADDING = 0

def get_inputs():
    """Returns a list of tensors to pass to forward()."""
    return [torch.randn(N, C, D, H, W)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__()."""
    return [KERNEL_SIZE, STRIDE, PADDING]