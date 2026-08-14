import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 2, conv)"""

    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels
        )
        
        # Pointwise (1x1) convolution
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Module-level constants for shapes
BATCH_SIZE = 3
IN_CHANNELS = 16
HEIGHT = 48
WIDTH = 48
KERNEL_SIZE = 3

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, KERNEL_SIZE]