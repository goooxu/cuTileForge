import torch
import torch.nn as nn

class Model(nn.Module):
    """GroupedConv3D (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, groups, kernel_size, 
                 stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        
        self.conv.eval()

    def forward(self, x):
        return self.conv(x)


IN_CHANNELS = 64
OUT_CHANNELS = 128
GROUPS = 8
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DILATION = 1
BIAS = True

BATCH_SIZE = 2
DEPTH = 16
HEIGHT = 32
WIDTH = 32


def get_init_inputs():
    return [
        IN_CHANNELS, 
        OUT_CHANNELS, 
        GROUPS, 
        KERNEL_SIZE, 
        STRIDE, 
        PADDING, 
        DILATION, 
        BIAS
    ]


def get_inputs():
    x = torch.ones(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH, 
                   dtype=torch.float32)
    return [x]