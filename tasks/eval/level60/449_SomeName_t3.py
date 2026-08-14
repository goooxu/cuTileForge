import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, pool)"""

    def __init__(self, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        # Create a buffer that will be used in forward pass for elementwise operation
        self.register_buffer('scale', torch.ones(1, 1, 1, 1))

    def forward(self, x):
        # Apply max pooling
        pooled = self.pool(x)
        # Elementwise multiplication with scale factor
        result = pooled * self.scale
        return result


# Module-level constants for shape configuration
BATCH_SIZE = 6
IN_CHANNELS = 256
HEIGHT = 384
WIDTH = 384
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [KERNEL_SIZE, STRIDE, PADDING]