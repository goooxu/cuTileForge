import torch
import torch.nn as nn

class Model(nn.Module):
    """ElementwiseChain (tier 5, elementwise)"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Chain of four elementwise operations
        y = x * 2.0
        y = y + 1.0
        y = torch.relu(y)
        y = y / 3.0
        return y

# Module-level constants for shape configuration
INPUT_HEIGHT = 1536
INPUT_WIDTH = 1536
BATCH_SIZE = 6
CHANNELS = 3

def get_inputs():
    """Return a list of input tensors for the forward pass."""
    # Create a large tensor suitable for throughput measurement
    return [torch.randn(BATCH_SIZE, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    # No arguments needed for this model
    return []