import torch
import torch.nn as nn


class Model(nn.Module):
    """ReductionWithElementwise (tier 5, reduction)"""

    def __init__(self, input_dim, reduction_dim):
        super().__init__()
        self.input_dim = input_dim
        self.reduction_dim = reduction_dim
        
        # Use BatchNorm but set to eval mode to ensure deterministic behavior
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.eval()
        
    def forward(self, x):
        # Ensure x is float32 for consistent operations
        x = x.float()
        
        # Step 1: Reduce along the last dimension (reduction along last axis)
        reduced = torch.sum(x, dim=-1)
        
        # Step 2: Apply elementwise work - multiply by scalar and add bias
        # Using stored module parameters for the elementwise operation
        scaled = reduced * 2.0 + 1.0
        
        return scaled


# Module-level constants for shape configuration
INPUT_DIM = 128
REDUCTION_DIM = 256

def get_inputs():
    """Return a list of tensors to pass to forward method."""
    # Create input tensor with shape (batch_size, input_dim, reduction_dim)
    # Using 16 as batch size for medium-sized computation
    batch_size = 16
    x = torch.randn(batch_size, INPUT_DIM, REDUCTION_DIM)
    return [x]

def get_init_inputs():
    """Return arguments to pass to __init__."""
    return [INPUT_DIM, REDUCTION_DIM]