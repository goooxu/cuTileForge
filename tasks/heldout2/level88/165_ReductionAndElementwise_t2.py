import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionAndElementwise (tier 2, reduction)"""

    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x):
        # Reduction along axis 1 (columns)
        reduced = torch.sum(x, dim=1)
        
        # Elementwise operations: multiply by 2 and add 1
        result = reduced * 2.0 + 1.0
        
        return result

# Module-level constants for shape
BATCH_SIZE = 128
INPUT_FEATURES = 4096

def get_inputs():
    # Return a list with one tensor for the forward pass
    # Large tensor suitable for measuring throughput
    return [torch.randn(BATCH_SIZE, INPUT_FEATURES)]

def get_init_inputs():
    # Return arguments for __init__ (empty list since __init__ takes no arguments)
    return []