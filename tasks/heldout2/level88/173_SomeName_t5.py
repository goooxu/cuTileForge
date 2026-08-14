import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(Model, self).__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
    
    def forward(self, x):
        return self.conv(x)

# Module-level constants for shapes
IN_CHANNELS = 16
OUT_CHANNELS = 32
KERNEL_SIZE = (3, 3, 3)
STRIDE = (2, 2, 2)
PADDING = (1, 1, 1)
DILATION = (1, 1, 1)

def get_inputs():
    batch_size = 4
    height = 16
    width = 16
    depth = 16
    x = torch.randn(batch_size, IN_CHANNELS, height, width, depth)
    return [x]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION]