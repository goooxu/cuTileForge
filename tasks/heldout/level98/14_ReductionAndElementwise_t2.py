import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionAndElementwise (tier 2, reduction)"""

    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.scale_param = nn.Parameter(torch.ones(1, input_dim))
        # Using BatchNorm to satisfy eval() requirement for determinism
        self.bn = nn.BatchNorm1d(input_dim)
        self.bn.eval()  # Set to eval mode for deterministic behavior

    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Reduce along dimension 1 (input_dim axis) to get shape (batch_size,)
        reduced = x.sum(dim=1)
        
        # Reshape reduced for broadcasting: (batch_size, 1)
        reduced_expanded = reduced.unsqueeze(-1)
        
        # Elementwise operations with input
        scaled_input = x * self.scale_param
        result = scaled_input + reduced_expanded
        
        return result

# Module-level constants for shape configuration
INPUT_DIM = 64
BATCH_SIZE = 32

def get_inputs():
    # Return list of input tensors for forward pass
    # Input tensor has shape (BATCH_SIZE, INPUT_DIM)
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    return [x]

def get_init_inputs():
    # Return list of arguments for __init__
    return [INPUT_DIM]