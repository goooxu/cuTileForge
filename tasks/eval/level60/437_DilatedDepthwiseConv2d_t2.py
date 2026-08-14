import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedDepthwiseConv2d (tier 2, conv)"""

    def __init__(self, in_channels=256, kernel_size=3, stride=1, padding=2, dilation=2):
        super(Model, self).__init__()
        
        # Depthwise convolution with dilation
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # depthwise: groups = input channels
            bias=False
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        return self.conv(x)


# Module-level constants for shapes
INPUT_BATCH_SIZE = 8
INPUT_CHANNELS = 256
INPUT_HEIGHT = 192
INPUT_WIDTH = 192
def get_inputs():
    # Return list with single tensor for input
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    # Return list of arguments for __init__
    return [256, 3, 1, 2, 2]  # in_channels=256, kernel_size=3, stride=1, padding=2, dilation=2