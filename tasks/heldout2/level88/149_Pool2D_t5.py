import torch
import torch.nn as nn

class Model(nn.Module):
    """Pool2D (tier 5, pool)"""

    def __init__(self, kernel_size, stride=None, padding=0):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)

# Module-level constants for shapes
KERNEL_SIZE = 2
STRIDE = 2
PADDING = 0
BATCH_SIZE = 4
CHANNELS = 3
HEIGHT = 32
WIDTH = 32

def get_inputs():
    """Return input tensors for the model."""
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model."""
    return [KERNEL_SIZE, STRIDE, PADDING]