import torch
import torch.nn as nn


class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.relu(out)
        out = out * 0.9 + 0.1
        return out


# Shape constants
IN_CHANNELS = 256
OUT_CHANNELS = 512
KERNEL_SIZE = 3
PADDING = 1
BATCH_SIZE = 32
HEIGHT = 128
WIDTH = 128

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING]

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]