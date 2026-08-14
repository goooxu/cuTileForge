import torch
import torch.nn as nn

class Model(nn.Module):
    """ChainElementwise (tier 2, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Chain of elementwise operations
        x = x * 2.0
        x = x + 1.0
        x = torch.relu(x)
        x = x / (x + 1.0)
        return x

# Module-level constants for tensor shapes
INPUT_SIZE = 10000000  # Large tensor for throughput measurement

def get_inputs():
    """Returns list of input tensors for forward pass"""
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    """Returns list of arguments for model initialization"""
    return []
_EVAL_MARK = 1
