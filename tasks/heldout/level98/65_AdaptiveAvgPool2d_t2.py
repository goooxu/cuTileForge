import torch
import torch.nn as nn

"""AdaptiveAvgPool2d (tier 2, pool)"""

class Model(nn.Module):
    """AdaptiveAvgPool2d (tier 2, pool)"""

    def __init__(self, output_size):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        self.output_size = output_size

    def forward(self, x):
        return self.pool(x)

# Module-level constants for shape configuration
INPUT_HEIGHT = 1024
INPUT_WIDTH = 1024
OUTPUT_SIZE = 64

def get_inputs():
    # Create input tensor of appropriate size
    # Shape: (batch, channels, height, width) = (1, 3, 1024, 1024)
    return [torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    # Return arguments for Model.__init__
    return [OUTPUT_SIZE]