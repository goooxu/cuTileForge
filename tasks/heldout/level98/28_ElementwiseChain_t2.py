import torch
import torch.nn as nn


class Model(nn.Module):
    """ElementwiseChain (tier 2, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Chain of 5 elementwise operations
        x = x * 2.0
        x = x + 1.0
        x = torch.relu(x)
        x = x / 3.0
        x = torch.tanh(x)
        return x


# Module-level constants for tensor shapes
INPUT_SIZE = [128, 64, 16, 16]

def get_inputs():
    """Returns a list of input tensors for the model forward pass."""
    # Create input tensor with deterministic values
    x = torch.ones(INPUT_SIZE)
    return [x]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return []