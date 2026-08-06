import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveMaxPool1D (tier 3, pool)"""
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.output_size = output_size
        # Create the pooling layer - it's stateless so no need for eval()
        self.pool = nn.AdaptiveMaxPool1d(output_size)
    
    def forward(self, x):
        return self.pool(x)

# Module-level constants for shape configuration
BATCH_SIZE = 32
IN_CHANNELS = 64
INPUT_LENGTH = 4096
OUTPUT_SIZE = 256

def get_inputs():
    """Returns a list with one tensor for the forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    """Returns arguments for the Model constructor"""
    return [OUTPUT_SIZE]