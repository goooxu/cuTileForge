import torch
import torch.nn as nn


class Model(nn.Module):
    """DepthwiseSeparableConv (tier 3, conv)"""

    def __init__(self, in_channels, depth_multiplier, kernel_size, input_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.out_channels = in_channels * depth_multiplier
        self.kernel_size = kernel_size
        self.input_size = input_size
        
        # Depthwise convolution: groups=in_channels, each input channel convolved independently
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * depth_multiplier,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution (1x1 convolution) to mix channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels * depth_multiplier,
            out_channels=in_channels * depth_multiplier,
            kernel_size=1,
            padding=0,
            groups=1,
            bias=False
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


# Configuration parameters
IN_CHANNELS = 256
DEPTH_MULTIPLIER = 2
KERNEL_SIZE = 3
INPUT_SIZE = (32, 256, 112, 112)  # batch_size, channels, height, width


def get_inputs():
    """Generate input tensors for the model"""
    batch_size = INPUT_SIZE[0]
    in_channels = IN_CHANNELS
    height = INPUT_SIZE[2]
    width = INPUT_SIZE[3]
    return [torch.randn(batch_size, in_channels, height, width)]


def get_init_inputs():
    """Generate initialization arguments for the model"""
    return [IN_CHANNELS, DEPTH_MULTIPLIER, KERNEL_SIZE, INPUT_SIZE]