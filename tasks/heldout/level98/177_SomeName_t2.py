import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""

# Module-level constants for tensor shapes
IN_CHANNELS = 16
OUT_CHANNELS = 16
KERNEL_SIZE = 3
BATCH_SIZE = 8
HEIGHT = 32
WIDTH = 32

class Model(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
                 kernel_size=KERNEL_SIZE, height=HEIGHT, width=WIDTH):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups = in_channels, each channel processed separately
        self.depthwise_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=False
        )
        
        # Initialize weights
        nn.init.kaiming_normal_(self.depthwise_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.pointwise_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # Depthwise convolution: applies convolution separately to each input channel
        depthwise = self.depthwise_conv(x)
        
        # Pointwise convolution: 1x1 convolution to combine channels
        pointwise = self.pointwise_conv(depthwise)
        
        return pointwise

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, HEIGHT, WIDTH]