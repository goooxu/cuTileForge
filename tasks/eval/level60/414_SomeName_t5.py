import torch
import torch.nn as nn

# Module-level constants for shapes
BATCH_SIZE = 12
IN_CHANNELS = 16
OUT_CHANNELS = 32
KERNEL_SIZE = 3
INPUT_HEIGHT = 48
INPUT_WIDTH = 48
class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups=in_channels, in_channels=1 per group
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            padding=kernel_size // 2
        )
        
        # Pointwise convolution: 1x1 convolution to mix channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
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

def get_inputs():
    """Return a list of tensors to pass to forward."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]