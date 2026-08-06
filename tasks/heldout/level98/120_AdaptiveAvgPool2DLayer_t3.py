import torch
import torch.nn as nn

"""AdaptiveAvgPool2DLayer (tier 3, pool)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 64
INPUT_HEIGHT = 16
INPUT_WIDTH = 16
OUTPUT_HEIGHT = 4
OUTPUT_WIDTH = 4

class Model(nn.Module):
    """AdaptiveAvgPool2DLayer (tier 3, pool)"""
    
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        # Using AdaptiveAvgPool2d which is an nn.Module and doesn't require eval()
    
    def forward(self, x):
        # Apply adaptive average pooling to the input tensor
        return nn.functional.adaptive_avg_pool2d(x, self.output_size)

def get_inputs():
    # Generate a tensor of shape [BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH]
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    # Return the arguments needed for __init__
    return [OUTPUT_HEIGHT]