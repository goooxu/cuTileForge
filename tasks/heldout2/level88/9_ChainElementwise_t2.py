import torch
import torch.nn as nn

class Model(nn.Module):
    """ChainElementwise (tier 2, elementwise)"""

    def __init__(self, input_size, num_channels):
        super(Model, self).__init__()
        self.input_size = input_size
        self.num_channels = num_channels

    def forward(self, x):
        # Chain of 4+ elementwise operations on the input tensor
        # 1. Square the input
        x = x * x
        
        # 2. Add a constant offset
        x = x + 1.0
        
        # 3. Take square root
        x = torch.sqrt(x)
        
        # 4. Multiply by a scale factor
        x = x * 2.5
        
        # 5. Apply exponential
        x = torch.exp(-x)
        
        return x


# Module-level constants for shape configuration
INPUT_SIZE = 4096
NUM_CHANNELS = 2048

def get_inputs():
    # Return a list with a single large tensor for the elementwise operations
    # Using float32 for compatibility and performance
    return [torch.randn(INPUT_SIZE, NUM_CHANNELS, dtype=torch.float32)]

def get_init_inputs():
    # Return the arguments needed for __init__
    return [INPUT_SIZE, NUM_CHANNELS]