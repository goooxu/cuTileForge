import torch
import torch.nn as nn

class Model(nn.Module):
    """ElementwiseChain (tier 3, elementwise)"""
    
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.input_size = input_size
    
    def forward(self, x):
        # Chain of elementwise operations
        x = x * 2.0
        x = x + 1.0
        x = x ** 2
        x = torch.sqrt(x)
        x = x - 1.0
        return x

# Module-level constants for shape configuration
INPUT_SIZE = 1000000  # Large tensor for throughput testing

def get_inputs():
    """Return list of input tensors for forward pass"""
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    """Return list of arguments for model initialization"""
    return [INPUT_SIZE]