import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveAvgPool1D (tier 5, pool)"""
    
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)
    
    def forward(self, x):
        return self.pool(x)

# Module-level constants for shapes
BATCH_SIZE = 4
INPUT_CHANNELS = 32
INPUT_LENGTH = 64
OUTPUT_SIZE = 16

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    return [OUTPUT_SIZE]