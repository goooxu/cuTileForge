import torch
import torch.nn as nn

"""AdaptiveAvgPool1DModel (tier 3, pool)"""

# Module-level constants for shapes
INPUT_BATCH = 4
INPUT_CHANNELS = 32
INPUT_LENGTH = 64
OUTPUT_LENGTH = 16

class Model(nn.Module):
    """AdaptiveAvgPool1DModel (tier 3, pool)"""
    
    def __init__(self, output_length):
        super(Model, self).__init__()
        self.output_length = output_length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_length)
    
    def forward(self, x):
        return self.adaptive_pool(x)

def get_inputs():
    # Create input tensor with shape (batch, channels, length)
    return [torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    # Return the output_length parameter for __init__
    return [OUTPUT_LENGTH]
_EVAL_MARK = 1
