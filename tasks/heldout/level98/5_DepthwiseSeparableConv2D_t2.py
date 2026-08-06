import torch
import torch.nn as nn

"""DepthwiseSeparableConv2D (tier 2, conv)"""


class DepthwiseSeparableConv2D(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier=1, bias=True):
        super(DepthwiseSeparableConv2D, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels * depth_multiplier, 
            kernel_size=self.kernel_size, 
            groups=in_channels, 
            bias=bias,
            padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier, 
            out_channels, 
            kernel_size=1, 
            bias=bias
        )
        
        self.depth_multiplier = depth_multiplier

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Model(nn.Module):
    """DepthwiseSeparableConv2D (tier 2, conv)"""

    def __init__(self, input_channels, output_channels, kernel_size, depth_multiplier):
        super(Model, self).__init__()
        
        self.conv = DepthwiseSeparableConv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            depth_multiplier=depth_multiplier,
            bias=True
        )

    def forward(self, x):
        return self.conv(x)


# Module-level constants for shapes
INPUT_BATCH_SIZE = 64
INPUT_CHANNELS = 128
OUTPUT_CHANNELS = 256
KERNEL_SIZE = 3
DEPTH_MULTIPLIER = 1

# Input tensor shape
INPUT_HEIGHT = 112
INPUT_WIDTH = 112


def get_inputs():
    """Return input tensors for the forward pass."""
    return [
        torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    ]


def get_init_inputs():
    """Return arguments for __init__."""
    return [
        INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, DEPTH_MULTIPLIER
    ]