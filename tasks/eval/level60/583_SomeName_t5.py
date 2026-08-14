import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Chain of 4 elementwise operations
        # 1. Square each element
        x = x * x
        
        # 2. Add constant offset
        x = x + 1.0
        
        # 3. Apply reciprocal
        x = 1.0 / x
        
        # 4. Apply exponential
        x = torch.exp(x)
        
        return x

# Module-level constants for shapes
INPUT_SHAPE = (64, 64, 32, 32)

def get_inputs():
    # Return list with single tensor for the model
    return [torch.randn(INPUT_SHAPE)]

def get_init_inputs():
    # Return empty list since __init__ doesn't take arguments
    return []
_EVAL_MARK = 1
