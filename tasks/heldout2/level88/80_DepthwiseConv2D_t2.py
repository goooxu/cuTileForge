import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseConv2D (tier 2, conv)"""
    
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, groups=None):
        super(Model, self).__init__()
        
        # If groups not specified, use in_channels for depthwise convolution
        if groups is None:
            groups = in_channels
        
        # Create depthwise convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        
        # Set to eval mode for deterministic behavior
        self.conv.eval()
    
    def forward(self, x):
        return self.conv(x)

# Module-level constants for tensor shapes
IN_CHANNELS = 16
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DILATION = 1
BATCH_SIZE = 8
HEIGHT = 32
WIDTH = 32

def get_inputs():
    # Create a batch of input tensors
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    # Return configuration for __init__
    return [IN_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION]