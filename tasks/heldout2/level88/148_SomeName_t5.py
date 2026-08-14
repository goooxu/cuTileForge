import torch
import torch.nn as nn

"""SomeName (tier 5, pool)"""

# Module-level constants for shape configuration
BATCH_SIZE = 2
IN_CHANNELS = 3
HEIGHT = 4
WIDTH = 4

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
    
    def forward(self, x):
        return self.pool(x)

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return []