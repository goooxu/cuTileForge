import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv3D (tier 2, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,
            bias=True
        )
    
    def forward(self, x):
        return self.conv(x)

# Module-level constants for shapes
IN_CHANNELS = 128
OUT_CHANNELS = 256
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 2
DILATION = 2

# Input shape for 3D convolution: (batch_size, channels, depth, height, width)
BATCH_SIZE = 8
DEPTH = 32
HEIGHT = 32
WIDTH = 32

def get_inputs():
    """Return input tensors for the model"""
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)
    ]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION]