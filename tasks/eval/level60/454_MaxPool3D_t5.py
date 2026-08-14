import torch
import torch.nn as nn

class Model(nn.Module):
    """MaxPool3D (tier 5, pool)"""
    
    def __init__(self, kernel_size, stride, padding, dilation):
        super(Model, self).__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=False,
            ceil_mode=False
        )
    
    def forward(self, x):
        return self.pool(x)

# Module-level constants for shape configuration
BATCH_SIZE = 12
CHANNELS = 64
DEPTH = 48
HEIGHT = 48
WIDTH = 48
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
DILATION = 1

def get_inputs():
    # Return a list with a single tensor for the model input
    return [torch.randn(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return arguments for __init__ method
    return [KERNEL_SIZE, STRIDE, PADDING, DILATION]