import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""

    def __init__(self, in_channels, groups, kernel_size=3):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.kernel_size = kernel_size
        
        # Compute output channels for depthwise convolution
        # Using same number of input channels for depthwise convolution
        depthwise_out_channels = in_channels
        
        # First convolution: depthwise convolution
        # This convolves each input channel separately
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=depthwise_out_channels,
            kernel_size=kernel_size,
            groups=in_channels,  # Each input channel convolved separately
            padding=1  # Preserve spatial dimensions
        )
        
        # Second convolution: pointwise convolution
        # This combines the depthwise features using 1x1 convolution
        self.pointwise_conv = nn.Conv2d(
            in_channels=depthwise_out_channels,
            out_channels=in_channels,  # Output same number of channels as input
            kernel_size=1,
            groups=1  # Standard 1x1 convolution
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Pointwise convolution
        x = self.pointwise_conv(x)
        
        return x

# Module-level constants for shapes
IN_CHANNELS = 8
GROUPS = 8
BATCH_SIZE = 1
HEIGHT = 32
WIDTH = 32
KERNEL_SIZE = 3

def get_inputs():
    """Return input tensor for the model"""
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    ]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [IN_CHANNELS, GROUPS, KERNEL_SIZE]