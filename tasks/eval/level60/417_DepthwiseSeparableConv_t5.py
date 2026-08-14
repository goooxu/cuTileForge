import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""

    def __init__(self, in_channels, depth_multiplier, kernel_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        
        # First convolution: depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels * depth_multiplier,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=kernel_size // 2
        )
        
        # Second convolution: pointwise convolution (1x1 conv)
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels * depth_multiplier,
            out_channels=in_channels,
            kernel_size=1
        )
        
        # Set to eval mode for deterministic behavior
        self.depthwise_conv.eval()
        self.pointwise_conv.eval()

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Pointwise convolution
        x = self.pointwise_conv(x)
        
        return x

# Module-level constants for shapes
IN_CHANNELS = 128
DEPTH_MULTIPLIER = 2
KERNEL_SIZE = 3
BATCH_SIZE = 2
HEIGHT = 336
WIDTH = 336
def get_inputs():
    """Return input tensors for the forward pass."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for __init__."""
    return [IN_CHANNELS, DEPTH_MULTIPLIER, KERNEL_SIZE]