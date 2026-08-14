import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv2D (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.dilated_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
    
    def forward(self, x):
        return self.dilated_conv(x)

# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3
DILATION = 2
BATCH_SIZE = 4
HEIGHT = 64
WIDTH = 64

def get_inputs():
    """Returns input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns initialization arguments for the model"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION]