import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, elementwise)"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y, z, w):
        # Chain of elementwise operations
        out = x + y
        out = out * z
        out = out - w
        out = torch.abs(out)
        out = torch.sqrt(out + 1e-6)
        return out

# Module-level constants for shapes
SHAPE = (32, 64)
BATCH_SIZE = 32
FEATURES = 64

def get_inputs():
    """Returns list of input tensors for forward pass"""
    return [
        torch.randn(SHAPE),
        torch.randn(SHAPE),
        torch.randn(SHAPE),
        torch.randn(SHAPE)
    ]

def get_init_inputs():
    """Returns list of arguments for __init__"""
    return []