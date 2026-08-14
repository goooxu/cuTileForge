import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveMaxPool1D (tier 2, pool)"""
    
    def __init__(self, output_size, input_size, channels):
        super(Model, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.channels = channels
        self.adaptive_pool = nn.AdaptiveMaxPool1d(output_size=output_size)
    
    def forward(self, x):
        return self.adaptive_pool(x)


# Module-level constants for shape configuration
OUTPUT_SIZE = 8
INPUT_SIZE = 32
CHANNELS = 16

def get_inputs():
    """Return input tensors for the model"""
    return [torch.randn(1, CHANNELS, INPUT_SIZE)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [OUTPUT_SIZE, INPUT_SIZE, CHANNELS]