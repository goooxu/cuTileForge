import torch
import torch.nn as nn

class Model(nn.Module):
    """ElementwiseChain (tier 5, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x):
        # Chain of four elementwise operations
        # 1. Add a constant
        x = x + 1.0
        # 2. Multiply by a constant
        x = x * 2.0
        # 3. Apply relu activation
        x = torch.relu(x)
        # 4. Apply tanh activation
        x = torch.tanh(x)
        return x

# Module-level constants for shapes
INPUT_SIZE = 1024
HIDDEN_SIZE = 2048
OUTPUT_SIZE = 512

def get_inputs():
    # Return a list with one large tensor for elementwise operations
    return [torch.randn(1024, 1024)]

def get_init_inputs():
    # No initialization arguments needed for this model
    return []