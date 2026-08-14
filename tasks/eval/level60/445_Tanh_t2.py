import torch
import torch.nn as nn

"""Tanh (tier 2, conv)"""

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups):
        super(Model, self).__init__()
        
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True
        )
        
        # Initialize weights with fixed values for deterministic behavior
        with torch.no_grad():
            nn.init.constant_(self.conv.weight, 0.1)
            nn.init.constant_(self.conv.bias, 0.0)
    
    def forward(self, x):
        return torch.tanh(self.conv(x))

# Module-level constants for tensor shapes
IN_CHANNELS = 2
OUT_CHANNELS = 4
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
DILATION = 1
GROUPS = 1
BATCH_SIZE = 2
HEIGHT = 6
WIDTH = 6
def get_inputs():
    """Return input tensor for the convolution"""
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """Return initialization parameters for the model"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION, GROUPS]