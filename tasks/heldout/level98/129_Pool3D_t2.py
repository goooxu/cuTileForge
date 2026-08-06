import torch
import torch.nn as nn

class Model(nn.Module):
    """Pool3D (tier 2, pool)"""

    def __init__(self, kernel_size=3, stride=2, padding=1, dilation=1):
        super(Model, self).__init__()
        # Using a non-trainable pooling layer, no BatchNorm needed, no evaluation state set
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
    
    def forward(self, x):
        # Ensure input is 5D tensor for 3D pooling
        return self.pool(x)


# Module-level constants for tensor shapes
BATCH_SIZE = 4
CHANNELS = 16
DEPTH = 64
HEIGHT = 64
WIDTH = 64

# Pooling layer configuration parameters
POOL_KERNEL_SIZE = 3
POOL_STRIDE = 2
POOL_PADDING = 1
POOL_DILATION = 1

def get_inputs():
    """Create a 5D input tensor for 3D pooling."""
    return [torch.randn(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return configuration parameters for the pooling layer initialization."""
    return [POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING, POOL_DILATION]