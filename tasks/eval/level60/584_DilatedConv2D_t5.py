import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv2D (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, bias=True)
        
    def forward(self, x):
        return self.conv(x)

# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3
DILATION = 2
BATCH_SIZE = 2
HEIGHT = 12
WIDTH = 12
def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION]