import torch
import torch.nn as nn

"""AdaptivePool2D (tier 2, pool)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 16
IN_CHANNELS = 256
INPUT_HEIGHT = 56
INPUT_WIDTH = 56
OUTPUT_HEIGHT = 7
OUTPUT_WIDTH = 7

class Model(nn.Module):
    """AdaptivePool2D (tier 2, pool)"""
    
    def __init__(self, output_size=(OUTPUT_HEIGHT, OUTPUT_WIDTH)):
        super(Model, self).__init__()
        self.output_size = output_size
        self.pool = nn.AdaptiveAvgPool2d(output_size)
    
    def forward(self, x):
        return self.pool(x)

def get_inputs():
    """Return input tensor for the model"""
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [(OUTPUT_HEIGHT, OUTPUT_WIDTH)]