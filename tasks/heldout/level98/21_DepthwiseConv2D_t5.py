import torch
import torch.nn as nn

"""DepthwiseConv2D (tier 5, conv)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 2
IN_CHANNELS = 4
OUT_CHANNELS = 8
HEIGHT = 6
WIDTH = 6
KERNEL_SIZE = 3

class Model(nn.Module):
    """DepthwiseConv2D (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Create depthwise convolution with groups=in_channels
        # Each input channel is processed independently
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,  # depthwise convolution
            bias=False,
            padding=0  # no padding to keep it simple
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()
        
    def forward(self, x):
        # Ensure deterministic behavior
        with torch.no_grad():
            return self.conv(x)

def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, height, width)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return configuration arguments for __init__
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]