import torch
import torch.nn as nn

"""ConvReLUPool (tier 5, conv)"""

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(pool_size)
        # Ensure deterministic behavior by setting to eval mode
        self.eval()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# Module-level constants for shapes
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
KERNEL_SIZE = 3
POOL_SIZE = 2
BATCH_SIZE = 32
HEIGHT = 224
WIDTH = 224


def get_inputs():
    # Create input tensor
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    return [x]


def get_init_inputs():
    # Return initialization arguments
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, POOL_SIZE]