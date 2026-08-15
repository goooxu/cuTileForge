import torch
import torch.nn as nn

class Model(nn.Module):
    """HardSwish (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
    
    def forward(self, x):
        return torch.nn.functional.hardswish(self.conv(x))

# Module-level constants for tensor shapes
IN_CHANNELS = 256
OUT_CHANNELS = 512
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DILATION = 2
GROUPS = 1
BATCH_SIZE = 13
HEIGHT = 337
WIDTH = 337
def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION, GROUPS]