import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveMaxPool1D (tier 5, pool)"""
    
    def __init__(self, output_size, input_length):
        super().__init__()
        self.output_size = output_size
        self.input_length = input_length
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size)
    
    def forward(self, x):
        return self.adaptive_pool(x)

# Module-level constants
OUTPUT_SIZE = 1024
INPUT_LENGTH = 4096
BATCH_SIZE = 8
IN_CHANNELS = 256

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    return [OUTPUT_SIZE, INPUT_LENGTH]