import torch
import torch.nn as nn

"""
Pool3D (tier 3, pool)
"""

# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 4
DEPTH = 16
HEIGHT = 16
WIDTH = 16
POOL_KERNEL_SIZE = (3, 3, 3)
POOL_STRIDE = (2, 2, 2)

class Model(nn.Module):
    """Pool3D (tier 3, pool)"""

    def __init__(self):
        super(Model, self).__init__()
        # Using nn.MaxPool3d for the pooling layer
        self.pool = nn.MaxPool3d(
            kernel_size=POOL_KERNEL_SIZE,
            stride=POOL_STRIDE,
            padding=0
        )
    
    def forward(self, x):
        # Ensure input has the correct shape: (batch, channels, depth, height, width)
        output = self.pool(x)
        return output

def get_inputs():
    """Returns a list of tensors to pass to forward."""
    # Create a tensor with shape (BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH, requires_grad=True)
    return [tensor]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return []