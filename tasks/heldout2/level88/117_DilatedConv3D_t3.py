import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv3D (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, 
            padding=dilation * (kernel_size - 1) // 2, 
            dilation=dilation
        )
        self.conv.eval()  # Ensure deterministic behavior
    
    def forward(self, x):
        return self.conv(x)

# Module-level constants for shapes
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
DILATION = 2
BATCH_SIZE = 8
DEPTH = 16
HEIGHT = 32
WIDTH = 32

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION]