import torch
import torch.nn as nn

class Model(nn.Module):
    """AvgPoolElementwise (tier 3, pool)"""

    def __init__(self, input_channels, output_channels, pool_size=2, pool_stride=2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride)
        # Initialize weight and bias for elementwise operation
        self.weight = nn.Parameter(torch.ones(output_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(output_channels, 1, 1))
        # Set to eval mode to ensure deterministic behavior
        self.eval()

    def forward(self, x):
        x = self.pool(x)
        # Elementwise multiplication and addition
        x = x * self.weight + self.bias
        return x


# Module-level constants for shapes
INPUT_BATCH = 2
INPUT_CHANNELS = 256
INPUT_HEIGHT = 512
INPUT_WIDTH = 512
OUTPUT_CHANNELS = 256
POOL_SIZE = 2
POOL_STRIDE = 2

def get_inputs():
    """Returns a list of input tensors for the model."""
    return [torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Returns arguments to pass to __init__."""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, POOL_SIZE, POOL_STRIDE]