import torch
import torch.nn as nn

"""AdaptiveAvgPool3D_1x3x7x7x7 (tier 2, pool)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 1
IN_CHANNELS = 3
DEPTH = 7
HEIGHT = 7
WIDTH = 7
OUTPUT_SIZE = (1, 1, 1)

class Model(nn.Module):
    """AdaptiveAvgPool3D_1x3x7x7x7 (tier 2, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(OUTPUT_SIZE)
    
    def forward(self, x):
        return self.pool(x)

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)]

def get_init_inputs():
    return []