import torch
import torch.nn as nn

"""SomeName (tier 5, elementwise)"""

class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""
    def __init__(self, input_size):
        super(Model, self).__init__()
        # These modules are placeholders to make the model non-trivial
        # but they won't affect the elementwise computation path
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        # Chain of elementwise operations
        # 1. Square operation
        out = x * x
        
        # 2. Add constant (broadcastable)
        out = out + 1.0
        
        # 3. Multiply by scalar
        out = out * 0.5
        
        # 4. Negative operation
        out = -out
        
        # 5. Reciprocal operation (1/x)
        out = 1.0 / out
        
        return out

# Module-level constants for shape
INPUT_SIZE = 16384
BATCH_SIZE = 64

def get_inputs():
    """Returns a list of tensors to pass to forward"""
    return [torch.randn(BATCH_SIZE, INPUT_SIZE)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return [INPUT_SIZE]