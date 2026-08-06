import torch
import torch.nn as nn

"""SomeName (tier 3, elementwise)"""

# Module-level constants for tensor shapes
INPUT_SIZE = 1024
HIDDEN_SIZE = 2048
OUTPUT_SIZE = 1024

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Chain of elementwise operations: x -> x^2 -> sqrt(|x|) -> sigmoid -> x * sigmoid(x)
        # All operations are elementwise, no in-place modifications
        x1 = x * x  # x^2
        x2 = torch.sqrt(torch.abs(x1))  # sqrt(|x^2|) = |x|
        x3 = torch.sigmoid(x2)  # sigmoid(|x|)
        x4 = x * x3  # x * sigmoid(|x|)
        return x4

def get_inputs():
    # Return a list containing a single input tensor
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    # No initialization arguments needed
    return []