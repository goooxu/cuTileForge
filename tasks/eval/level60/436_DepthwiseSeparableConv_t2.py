import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Model, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Module-level constants for shapes
IN_CHANNELS = 64
OUT_CHANNELS = 128
BATCH_SIZE = 7
HEIGHT = 49
WIDTH = 49
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

def get_inputs():
    # Create a sample input tensor for the model
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return arguments to pass to __init__
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]