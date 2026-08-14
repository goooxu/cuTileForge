import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveAvgPool1D (tier 2, pool)"""
    
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
    
    def forward(self, x):
        return self.pool(x)

# Module-level constants for shapes
INPUT_BATCH_SIZE = 4
INPUT_CHANNELS = 16
INPUT_LENGTH = 64
OUTPUT_LENGTH = 8

def get_inputs():
    """Return a list of tensors to pass to forward."""
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [OUTPUT_LENGTH]