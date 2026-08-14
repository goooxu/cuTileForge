import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 128
OUT_CHANNELS = 128
HEIGHT = 112
WIDTH = 112
KERNEL_SIZE = 3
PADDING = 1

class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Depthwise convolution: groups=in_channels, in_channels per group
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels
        )
        
        # Pointwise convolution: 1x1 convolution to mix channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0
        )

    def forward(self, x):
        # Depthwise convolution
        depthwise_out = self.depthwise_conv(x)
        # Pointwise convolution
        pointwise_out = self.pointwise_conv(depthwise_out)
        return pointwise_out

def get_inputs():
    # Return input tensor with appropriate shape for large tensor testing
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return initialization parameters for the model
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING]