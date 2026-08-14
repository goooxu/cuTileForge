import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 3, conv)"""
    
    def __init__(self, in_channels=4, kernel_size=3, depth_multiplier=1):
        super(Model, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.depth_multiplier = depth_multiplier
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=kernel_size // 2
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels * depth_multiplier,
            in_channels,
            kernel_size=1
        )
        
        # Set to eval mode for deterministic behavior
        self.depthwise.eval()
        self.pointwise.eval()

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise(x)
        # Pointwise convolution
        x = self.pointwise(x)
        return x


# Module-level constants for shapes
IN_CHANNELS = 4
KERNEL_SIZE = 3
DEPTH_MULTIPLIER = 1
BATCH_SIZE = 3
HEIGHT = 12
WIDTH = 12
def get_inputs():
    """Return list of input tensors."""
    # Create a 4D tensor for batch of images: (batch, channels, height, width)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return list of arguments for __init__."""
    return [IN_CHANNELS, KERNEL_SIZE, DEPTH_MULTIPLIER]