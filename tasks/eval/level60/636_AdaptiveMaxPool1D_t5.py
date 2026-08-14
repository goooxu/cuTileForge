import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveMaxPool1D (tier 5, pool)"""
    
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveMaxPool1d(output_size)
    
    def forward(self, x):
        return self.pool(x)

INPUT_SIZE = 1024
BATCH_SIZE = 96
CHANNELS = 256
OUTPUT_SIZE = 128

def get_inputs():
    x = torch.randn(BATCH_SIZE, CHANNELS, INPUT_SIZE)
    return [x]

def get_init_inputs():
    return [OUTPUT_SIZE]