import torch
import torch.nn as nn

class DepthwiseConv2d(nn.Conv2d):
    """Depthwise convolution layer."""
    
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, groups=None, bias=False):
        groups = in_channels if groups is None else groups
        super().__init__(in_channels, in_channels, kernel_size, stride=stride, 
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

class PointwiseConv2d(nn.Conv2d):
    """Pointwise (1x1) convolution layer."""
    
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias)

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution (tier 2, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = DepthwiseConv2d(in_channels, kernel_size=kernel_size, 
                                         stride=stride, padding=padding, groups=in_channels)
        self.pointwise = PointwiseConv2d(in_channels, out_channels, stride=1)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Model(nn.Module):
    """DepthwiseSeparableConvModel (tier 2, conv)"""
    
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = DepthwiseSeparableConv(input_channels, output_channels, 
                                          kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        return self.conv(x)

# Module-level constants for shapes
BATCH_SIZE = 4
INPUT_CHANNELS = 32
OUTPUT_CHANNELS = 64
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
INPUT_HEIGHT = 64
INPUT_WIDTH = 64

def get_inputs():
    """Return input tensors for the model."""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization."""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]