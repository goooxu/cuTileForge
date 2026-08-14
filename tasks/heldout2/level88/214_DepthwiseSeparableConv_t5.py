import torch
import torch.nn as nn

# Module-level constants for tensor shapes
INPUT_CHANNELS = 16
OUTPUT_CHANNELS = 32
KERNEL_SIZE = 3
BATCH_SIZE = 4
HEIGHT = 32
WIDTH = 32

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups=in_channels, input channels = in_channels
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels
        )
        
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        # Pointwise convolution
        x = self.pointwise_conv(x)
        return x

def get_inputs():
    """Return input tensors for the model"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE]