import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 2, conv)"""
    
    def __init__(self, in_channels, depth_multiplier, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups=in_channels, each input channel convolved independently
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels * depth_multiplier, 
            kernel_size=kernel_size, 
            groups=in_channels, 
            bias=False
        )
        
        # Pointwise convolution: 1x1 conv to combine channels
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier, 
            in_channels * depth_multiplier, 
            kernel_size=1, 
            bias=False
        )
        
        # Set to eval mode for deterministic behavior
        self.depthwise.eval()
        self.pointwise.eval()
    
    def forward(self, x):
        # Depthwise convolution
        out = self.depthwise(x)
        # Pointwise convolution
        out = self.pointwise(out)
        return out

# Module-level constants for shapes
IN_CHANNELS = 64
DEPTH_MULTIPLIER = 2
KERNEL_SIZE = 3
BATCH_SIZE = 1
HEIGHT = 256
WIDTH = 256

def get_inputs():
    """Return input tensor for forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [IN_CHANNELS, DEPTH_MULTIPLIER, KERNEL_SIZE]