import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 2, conv)"""
    
    def __init__(self, in_channels, kernel_size=3, depth_multiplier=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups=in_channels to process each channel separately
        self.depthwise_conv = nn.Conv2d(
            in_channels, 
            in_channels * depth_multiplier, 
            kernel_size=kernel_size, 
            groups=in_channels, 
            bias=bias
        )
        
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise_conv = nn.Conv2d(
            in_channels * depth_multiplier,
            in_channels,
            kernel_size=1,
            bias=bias
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()
    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 64
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

def get_inputs():
    """Return input tensor for forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [IN_CHANNELS]