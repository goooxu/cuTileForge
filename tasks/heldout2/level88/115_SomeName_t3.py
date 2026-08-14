import torch
import torch.nn as nn

"""SomeName (tier 3, elementwise)"""

class Model(nn.Module):
    """SomeName (tier 3, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Chain of four elementwise operations
        x = x + 1.0
        x = x * 2.0
        x = torch.relu(x)
        x = x - 0.5
        return x

# Module-level constants for shape
INPUT_SIZE = 100000000

def get_inputs():
    # Return a list with a single tensor for the elementwise operations
    x = torch.randn(INPUT_SIZE)
    return [x]

def get_init_inputs():
    # No arguments needed for initialization
    return []