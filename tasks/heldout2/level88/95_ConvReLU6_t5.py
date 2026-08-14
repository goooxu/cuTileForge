import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvReLU6 (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu6(x)
        return x

IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3

def get_inputs():
    return [torch.randn(1, IN_CHANNELS, 32, 32)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]