import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Model, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels,
            bias=False
        )
        
        # Pointwise (1x1) convolution
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Module-level constants for shape configuration
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
BATCH_SIZE = 32
HEIGHT = 256
WIDTH = 256

def get_inputs():
    # Create input tensor with consistent shape
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    # Return arguments for __init__
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]