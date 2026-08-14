import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionSum (tier 3, reduction)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Reduction along axis 1 followed by elementwise operations
        reduced = x.sum(dim=1)
        result = reduced * 2.0 + 1.0
        return result

# Module-level constants for shape configuration
INPUT_DIM_0 = 1024
INPUT_DIM_1 = 2048
INPUT_DIM_2 = 128

def get_inputs():
    # Create a large tensor for reduction operation
    # Size: (INPUT_DIM_0, INPUT_DIM_1, INPUT_DIM_2)
    # This is designed for throughput measurement
    return [torch.randn(INPUT_DIM_0, INPUT_DIM_1, INPUT_DIM_2)]

def get_init_inputs():
    # No initialization parameters needed
    return []