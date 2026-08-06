import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionModel (tier 2, reduction)"""

    def __init__(self, input_dim, output_dim, reduction_dim):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.reduction_dim = reduction_dim
        
        # Linear layer to project reduced features
        self.proj = nn.Linear(input_dim, output_dim, bias=False)
        
        # Use batch norm for post-processing, set to eval mode for determinism
        self.bn = nn.BatchNorm1d(output_dim)
        self.bn.eval()

    def forward(self, x):
        # x: (batch_size, reduction_dim, input_dim)
        # Reduce along reduction_dim (axis 1)
        reduced = x.mean(dim=1)  # (batch_size, input_dim)
        
        # Project reduced features
        projected = self.proj(reduced)  # (batch_size, output_dim)
        
        # Apply batch norm (deterministic in eval mode)
        output = self.bn(projected)  # (batch_size, output_dim)
        
        return output

# Module-level constants for shapes
INPUT_DIM = 64
OUTPUT_DIM = 128
REDUCTION_DIM = 16
BATCH_SIZE = 32

def get_inputs():
    # Return input tensor of shape (batch_size, reduction_dim, input_dim)
    return [torch.randn(BATCH_SIZE, REDUCTION_DIM, INPUT_DIM)]

def get_init_inputs():
    # Return arguments for __init__: input_dim, output_dim, reduction_dim
    return [INPUT_DIM, OUTPUT_DIM, REDUCTION_DIM]