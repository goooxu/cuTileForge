import torch
import torch.nn as nn

"""SomeName (tier 2, elementwise)"""

# Module-level constants for shape configuration
INPUT_SIZE = 1000000
HIDDEN_SIZE = 4000000

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Chain of elementwise operations for throughput testing
        # Operation 1: Square the input
        x = x * x
        
        # Operation 2: Add constant
        x = x + 1.0
        
        # Operation 3: Take reciprocal
        x = 1.0 / x
        
        # Operation 4: Square root
        x = torch.sqrt(x)
        
        # Operation 5: Exponential
        x = torch.exp(-x)
        
        # Operation 6: Logarithm
        x = torch.log(1.0 + x)
        
        return x

def get_inputs():
    # Return a list with a single large tensor for elementwise operations
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    # Return empty list since __init__ takes no arguments
    return []