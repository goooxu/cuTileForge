import torch
import torch.nn as nn

"""SomeName (tier 2, pool)"""

class Model(nn.Module):
    """SomeName (tier 2, pool)"""
    
    def __init__(self, kernel_size, stride, padding, channels):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.channels = channels
        # Using AveragePool2d for deterministic behavior
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=False
        )
        # Ensure deterministic forward pass
        self.eval()
    
    def forward(self, x):
        # Apply average pooling to the input tensor
        return self.pool(x)


# Module-level constants for tensor shapes
BATCH_SIZE = 8
CHANNELS = 32
HEIGHT = 64
WIDTH = 64

def get_inputs():
    """Return a list of tensors to pass to forward."""
    # Create input tensor with shape (batch, channels, height, width)
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [3, 2, 1, CHANNELS]