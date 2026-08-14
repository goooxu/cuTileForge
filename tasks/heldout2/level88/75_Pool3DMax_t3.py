import torch
import torch.nn as nn

"""Pool3DMax (tier 3, pool)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 8
CHANNELS = 32
DEPTH = 64
HEIGHT = 64
WIDTH = 64
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

class Model(nn.Module):
    """Pool3DMax (tier 3, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Create a 3D max pooling layer
        self.pool = nn.MaxPool3d(
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING
        )
    
    def forward(self, x):
        # Apply 3D max pooling
        return self.pool(x)

def get_inputs():
    # Create input tensor with the specified shape
    input_tensor = torch.randn(BATCH_SIZE, CHANNELS, DEPTH, HEIGHT, WIDTH)
    return [input_tensor]

def get_init_inputs():
    # No additional initialization inputs needed
    return []