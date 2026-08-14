import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 4
CHANNELS = 8
HEIGHT = 16
WIDTH = 16

class Model(nn.Module):
    """SomeName (tier 2, reduction)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Reduction along the height axis (axis 2)
        reduced = x.sum(dim=2)  # Shape: (BATCH_SIZE, CHANNELS, WIDTH)
        
        # Elementwise operation: multiply by a scalar factor
        result = reduced * 0.5
        
        return result

def get_inputs():
    # Create input tensor with shape (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # No initialization arguments needed
    return []