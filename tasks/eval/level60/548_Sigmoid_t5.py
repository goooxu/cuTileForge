import torch
import torch.nn as nn

"""Sigmoid (tier 5, pool)"""

# Module-level constants for shape configuration
BATCH_SIZE = 3
IN_CHANNELS = 3
HEIGHT = 6
WIDTH = 6
class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
    
    def forward(self, x):
        return torch.sigmoid(self.pool(x))

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return []