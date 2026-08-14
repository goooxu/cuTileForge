import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, reduction)"""
    
    def __init__(self, input_size, dim_to_reduce):
        super(Model, self).__init__()
        self.dim_to_reduce = dim_to_reduce
        self.input_size = input_size
        
    def forward(self, x):
        # Reduce along specified axis
        reduced = x.sum(dim=self.dim_to_reduce)
        
        # Elementwise operations
        result = reduced * 2.0 + 1.0
        
        return result

# Module-level constants for shapes
INPUT_SIZE = (4, 5, 6)
DIM_TO_REDUCE = 1

def get_inputs():
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    return [INPUT_SIZE, DIM_TO_REDUCE]