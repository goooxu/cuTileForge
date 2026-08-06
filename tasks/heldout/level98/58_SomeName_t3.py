import torch
import torch.nn as nn

"""
SomeName (tier 3, reduction)
"""

# Shape constants
BATCH_SIZE = 4
CHANNELS = 16
HEIGHT = 32
WIDTH = 32

class Model(nn.Module):
    """SomeName (tier 3, reduction)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # No learnable parameters needed for this reduction operation
        
    def forward(self, x):
        # Reduction: compute mean across the channel dimension
        reduced = x.mean(dim=1, keepdim=True)
        
        # Elementwise operation: apply a learnable scaling factor and add bias
        # Since we need learnable parameters, we'll create them here
        scale = nn.Parameter(torch.ones(1, 1, 1, 1))
        bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
        
        # Apply scale and bias (elementwise operations)
        output = reduced * scale + bias
        
        return output

def get_inputs():
    # Return a list with a single tensor of the appropriate shape
    # Shape: (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return an empty list since __init__ takes no arguments
    return []