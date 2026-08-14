import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

# Module-level constants for shape configuration
BATCH_SIZE = 8
IN_CHANNELS = 64
TIME_DIM = 64
HEIGHT = 64
WIDTH = 64
KERNEL_SIZE = 3
STRIDE = 2

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Create a 3D max pooling layer
        self.pool = nn.MaxPool3d(
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=0
        )
    
    def forward(self, x):
        # Apply 3D max pooling
        return self.pool(x)

def get_inputs():
    # Create input tensor with shape (BATCH_SIZE, IN_CHANNELS, TIME_DIM, HEIGHT, WIDTH)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, TIME_DIM, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return empty list since __init__ doesn't take arguments
    return []