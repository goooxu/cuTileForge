import torch
import torch.nn as nn

"""SomeName (tier 5, reduction)"""

# Module-level constants for tensor shapes
INPUT_SIZE = 4096
REDUCTION_DIM = 1

class Model(nn.Module):
    """SomeName (tier 5, reduction)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Reduction along axis 1: compute sum along reduction dimension
        reduced = x.sum(dim=REDUCTION_DIM)
        
        # Elementwise operation: multiply by 2 and add 1
        result = reduced * 2.0 + 1.0
        
        return result

def get_inputs():
    # Create large tensor suitable for throughput measurement
    # Shape: (batch_size, reduction_dim, other_dims)
    batch_size = 1024
    shape = (batch_size, REDUCTION_DIM, INPUT_SIZE // REDUCTION_DIM)
    return [torch.randn(*shape)]

def get_init_inputs():
    return []