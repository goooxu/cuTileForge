import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvReLU6 (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu6(x)
        return x


# Module-level constants for shape configuration
IN_CHANNELS = 256
OUT_CHANNELS = 512
KERNEL_SIZE = 3
PADDING = 1
BATCH_SIZE = 64
HEIGHT = 128
WIDTH = 128

def get_inputs():
    """Return input tensors for the forward pass."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments to pass to __init__."""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING]