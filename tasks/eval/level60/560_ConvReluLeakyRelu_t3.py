import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvReluLeakyRelu (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, leaky_negative_slope):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_negative_slope)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.leaky_relu(x)
        return x

# Module-level constants for shapes
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
LEAKY_NEGATIVE_SLOPE = 0.2
BATCH_SIZE = 12
HEIGHT = 48
WIDTH = 49
def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, LEAKY_NEGATIVE_SLOPE]