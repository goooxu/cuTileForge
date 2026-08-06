import torch
import torch.nn as nn

"""
DepthwiseSeparableConv (tier 2, conv)
"""

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 8
OUT_CHANNELS = 16
HEIGHT = 32
WIDTH = 32
KERNEL_SIZE = 3
PADDING = 1
STRIDE = 1

class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(Model, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        
        # First convolution: depthwise convolution (groups = in_channels)
        self.depthwise_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels,
            bias=False
        )
        
        # Second convolution: pointwise convolution (1x1 conv to expand channels)
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        
        # Ensure deterministic behavior
        self.eval()

    def forward(self, x):
        # First: depthwise convolution
        x = self.depthwise_conv(x)
        
        # Second: pointwise convolution
        x = self.pointwise_conv(x)
        
        return x


def get_inputs():
    """Return a list of tensors to pass to forward method."""
    # Create input tensor with shape (batch_size, in_channels, height, width)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [IN_CHANNELS, OUT_CHANNELS]