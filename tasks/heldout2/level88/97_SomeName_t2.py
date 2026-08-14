import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Chain of 4 elementwise operations on input tensor
        # Operation 1: Add constant
        x = x + 1.0
        
        # Operation 2: Multiply by constant
        x = x * 2.0
        
        # Operation 3: Apply ReLU
        x = torch.relu(x)
        
        # Operation 4: Square the values
        x = x * x
        
        return x

# Module-level constants for shapes
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
INPUT_CHANNELS = 3

def get_inputs():
    # Return a list with a single tensor of appropriate size
    return [torch.randn(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)]

def get_init_inputs():
    # Return arguments for __init__ (empty list since no args needed)
    return []