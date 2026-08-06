import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 2, conv)"""
    def __init__(self, in_channels, groups, kernel_size, stride=1, padding=0, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=groups,
            bias=bias
        )
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            groups=1, 
            bias=bias
        )
        
    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise(x)
        # Pointwise convolution
        x = self.pointwise(x)
        return x


# Module-level constants for shapes
INPUT_CHANNELS = 64
GROUPS = 64
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
BATCH_SIZE = 1
HEIGHT = 64
WIDTH = 64

def get_inputs():
    """Returns list of input tensors for forward pass"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns list of arguments for __init__"""
    return [INPUT_CHANNELS, GROUPS, KERNEL_SIZE, STRIDE, PADDING, False]