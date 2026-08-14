import torch
import torch.nn as nn

"""SomeName (tier 2, elementwise)"""

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Chain of four elementwise operations
        x = x * 2.0
        x = x + 1.0
        x = torch.abs(x)
        x = x / (1.0 + torch.abs(x))
        return x

# Module-level constants for shape
INPUT_SIZE = 1000

def get_inputs():
    """Return input tensor for the model"""
    return [torch.ones(INPUT_SIZE)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return []