import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv3D (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
        )
        # Initialize weights deterministically
        nn.init.kaiming_uniform_(self.conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x):
        return self.conv(x)

# Module-level constants for tensor shapes
IN_CHANNELS = 16
OUT_CHANNELS = 32
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DILATION = 2

INPUT_BATCH_SIZE = 4
INPUT_DEPTH = 8
INPUT_HEIGHT = 16
INPUT_WIDTH = 32

def get_inputs():
    return [torch.randn(INPUT_BATCH_SIZE, IN_CHANNELS, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION]