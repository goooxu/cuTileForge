import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, reduction)"""
    
    def __init__(self, input_size, reduction_dim):
        super().__init__()
        self.input_size = input_size
        self.reduction_dim = reduction_dim
        
    def forward(self, x):
        # Reduction along the specified dimension
        reduced = x.sum(dim=self.reduction_dim, keepdim=True)
        
        # Elementwise operations on the reduced tensor
        result = reduced * 2.0 + 1.0
        
        return result

# Module-level constants for shapes
INPUT_SIZE = [3, 4, 5, 6]
REDUCTION_DIM = 3
def get_inputs():
    """Returns a list of tensors to pass to forward"""
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return [INPUT_SIZE, REDUCTION_DIM]