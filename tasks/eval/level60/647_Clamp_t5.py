import torch
import torch.nn as nn

class Model(nn.Module):
    """Clamp (tier 5, pool)"""
    
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.output_size = output_size
    
    def forward(self, x):
        return torch.clamp(nn.functional.adaptive_avg_pool1d(x, self.output_size), min=-1.0, max=1.0)

# Module-level constants for shapes
INPUT_BATCH_SIZE = 2
INPUT_CHANNELS = 3
INPUT_LENGTH = 7
OUTPUT_SIZE = 4

def get_inputs():
    """Returns input tensor for the model"""
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    """Returns initialization arguments for the model"""
    return [OUTPUT_SIZE]