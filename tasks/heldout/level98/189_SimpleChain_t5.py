import torch
import torch.nn as nn

"""
SimpleChain (tier 5, elementwise)
"""

# Module-level constants for shape definitions
N, C, H, W = 2, 3, 4, 5
INPUT_SIZE = (N, C, H, W)

class Model(nn.Module):
    """SimpleChain (tier 5, elementwise)"""

    def __init__(self):
        super().__init__()
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Chain of 4+ elementwise operations
        x = self.activation(x)
        x = x * 2.0
        x = x + 1.0
        x = torch.pow(x, 2.0)
        return x

def get_inputs():
    # Generate random input within reasonable range to avoid extreme values in the chain
    return [torch.randn(INPUT_SIZE) * 0.5]

def get_init_inputs():
    # No additional initialization arguments needed
    return []