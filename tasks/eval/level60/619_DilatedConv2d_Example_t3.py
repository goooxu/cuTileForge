import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv2d_Example (tier 3, conv)"""

    def __init__(self, in_channels=2, out_channels=4, kernel_size=3, dilation=2):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Create a dilated convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True
        )
        
        # Initialize weights deterministically
        nn.init.constant_(self.conv.weight, 0.1)
        nn.init.constant_(self.conv.bias, 0.01)

    def forward(self, x):
        return self.conv(x)


# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 2
OUT_CHANNELS = 4
KERNEL_SIZE = 3
DILATION = 2
INPUT_HEIGHT = 12
INPUT_WIDTH = 12
def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]


def get_init_inputs():
    """Return arguments for model initialization"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION]