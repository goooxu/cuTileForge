import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Model, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                        stride=stride, padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.eval()  # Set to eval mode for deterministic behavior

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# Module-level constants for shapes
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
BATCH_SIZE = 4
HEIGHT = 32
WIDTH = 32

def get_inputs():
    """Return list of tensors to pass to forward"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return list of arguments to pass to __init__"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]