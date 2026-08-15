import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv2d (tier 2, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=groups, 
            bias=bias
        )
        
    def forward(self, x):
        return self.conv(x)

# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 64
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 2
DILATION = 2
GROUPS = 1
BATCH_SIZE = 7
HEIGHT = 49
WIDTH = 49
def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, height, width)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return arguments for __init__ using the module-level constants
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION, GROUPS, True]