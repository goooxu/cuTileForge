import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""

    def __init__(self, in_channels, depthwise_multiplier, kernel_size=3, padding=1):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.depthwise_multiplier = depthwise_multiplier
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * depthwise_multiplier,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels * depthwise_multiplier,
            in_channels * depthwise_multiplier,
            kernel_size=1
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise(x)
        # Pointwise convolution
        x = self.pointwise(x)
        return x


# Module-level constants for shapes
BATCH_SIZE = 24
IN_CHANNELS = 128
DEPTHWISE_MULTIPLIER = 2
KERNEL_SIZE = 3
PADDING = 1
HEIGHT = 192
WIDTH = 192
def get_inputs():
    """Return input tensor for forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    """Return arguments for __init__"""
    return [IN_CHANNELS, DEPTHWISE_MULTIPLIER, KERNEL_SIZE, PADDING]