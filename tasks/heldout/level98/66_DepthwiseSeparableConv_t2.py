import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 2, conv)"""
    
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # Initialize with constant values for deterministic behavior
        with torch.no_grad():
            nn.init.constant_(self.depthwise.weight, 0.5)
            nn.init.constant_(self.pointwise.weight, 0.5)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Module-level constants
BATCH_SIZE = 1
IN_CHANNELS = 2
HEIGHT = 4
WIDTH = 4
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

def get_inputs():
    """Return input tensors for forward pass."""
    x = torch.ones(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """Return arguments for __init__."""
    return [IN_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]