import torch
import torch.nn as nn

"""SiLU (tier 5, pool)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 12
IN_CHANNELS = 64
DEPTH = 24
HEIGHT = 48
WIDTH = 48
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

class Model(nn.Module):
    """Pool3DLayer (tier 5, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=PADDING
        )
    
    def forward(self, x):
        return torch.nn.functional.silu(self.pool(x))

def get_inputs():
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    ]

def get_init_inputs():
    return []