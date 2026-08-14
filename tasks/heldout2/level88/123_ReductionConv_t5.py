import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionConv (tier 5, reduction)"""
    
    def __init__(self, in_channels, reduction_dim, output_dim):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.reduction_dim = reduction_dim
        self.output_dim = output_dim
        
        # Linear layer for elementwise work after reduction
        self.elementwise_layer = nn.Linear(reduction_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, in_channels)
        
        # Reduction along sequence_length axis (dim=1) - compute mean
        reduced = x.mean(dim=1)  # Shape: (batch_size, in_channels)
        
        # Elementwise work - linear transformation
        output = self.elementwise_layer(reduced)  # Shape: (batch_size, output_dim)
        
        return output


# Module-level constants for tensor shapes
BATCH_SIZE = 32
SEQUENCE_LENGTH = 1024
IN_CHANNELS = 512
REDUCTION_DIM = 512
OUTPUT_DIM = 1024

def get_inputs():
    # Create input tensor with shape (batch_size, sequence_length, in_channels)
    x = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, IN_CHANNELS)
    return [x]

def get_init_inputs():
    # Return arguments for model initialization
    return [IN_CHANNELS, REDUCTION_DIM, OUTPUT_DIM]