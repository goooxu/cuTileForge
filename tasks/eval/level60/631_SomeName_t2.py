import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""

# Module-level constants for shape configuration
BATCH_SIZE = 3
IN_CHANNELS = 3
OUT_CHANNELS = 6
KERNEL_SIZE = 3
INPUT_HEIGHT = 12
INPUT_WIDTH = 12
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True
        )
        
    def forward(self, x):
        return self.conv(x)

def get_inputs():
    """Return list of input tensors for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return list of arguments for model initialization"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]