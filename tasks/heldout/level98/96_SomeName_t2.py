import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""
class Model(nn.Module):
    def __init__(self, in_channels, depth_multiplier, kernel_size, padding):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Depthwise convolution: groups=in_channels applies conv to each channel separately
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * depth_multiplier,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=padding,
            bias=False
        )
        
        # Pointwise convolution: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(
            in_channels=in_channels * depth_multiplier,
            out_channels=in_channels * depth_multiplier,
            kernel_size=1,
            groups=1,
            bias=False
        )
        
        # Set to eval mode for deterministic behavior
        self.depthwise.eval()
        self.pointwise.eval()

    def forward(self, x):
        # Depthwise convolution: applies conv to each input channel separately
        x = self.depthwise(x)
        # Pointwise convolution: 1x1 conv to combine channels
        x = self.pointwise(x)
        return x

# Module-level constants for shapes
INPUT_BATCH_SIZE = 4
INPUT_CHANNELS = 16
INPUT_HEIGHT = 32
INPUT_WIDTH = 32
DEPTH_MULTIPLIER = 2
KERNEL_SIZE = 3
PADDING = 1

def get_inputs():
    """Returns a list of tensors to pass to forward."""
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return [INPUT_CHANNELS, DEPTH_MULTIPLIER, KERNEL_SIZE, PADDING]