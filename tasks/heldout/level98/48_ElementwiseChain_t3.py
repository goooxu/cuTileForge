import torch
import torch.nn as nn

class Model(nn.Module):
    """ElementwiseChain (tier 3, elementwise)"""

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        # Chain of four elementwise operations on large tensor
        x = x * 2.0
        x = torch.relu(x)
        x = x + 1.0
        x = torch.sigmoid(x)
        return x

# Module-level constants for shape
BATCH_SIZE = 1024
FEATURES = 1024

def get_inputs():
    """Returns a list containing one large input tensor"""
    return [torch.randn(BATCH_SIZE, FEATURES)]

def get_init_inputs():
    """Returns a list of arguments for Model initialization"""
    return [FEATURES]