import torch
import torch.nn as nn

"""
DSConvSmall (tier 3, conv)
"""

# Module-level constants for shapes
INPUT_CHANNELS = 4
KERNEL_SIZE = 3
INPUT_BATCH_SIZE = 2
INPUT_HEIGHT = 6
INPUT_WIDTH = 6

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self, in_channels=INPUT_CHANNELS, kernel_size=KERNEL_SIZE):
        super(Model, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups=in_channels, out_channels=in_channels
        # Input: (B, C, H, W), Output: (B, C, H, W)
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=kernel_size // 2,
            bias=False
        )
        
        # Pointwise convolution (1x1 conv): reduces channels back to 1 or keeps same
        # Input: (B, C, H, W), Output: (B, 1, H, W) or (B, C, H, W) depending on design
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            bias=False
        )
    
    def forward(self, x):
        # Depthwise convolution: applies each filter to one input channel
        x = self.depthwise_conv(x)
        
        # Pointwise convolution: linear combinations of channels
        x = self.pointwise_conv(x)
        
        return x


def get_inputs():
    """Returns a list of tensors to pass to forward"""
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]


def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return []