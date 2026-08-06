import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self, in_channels, kernel_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                        groups=in_channels, bias=False)
        # Pointwise convolution
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        # Apply depthwise convolution
        x = self.depthwise_conv(x)
        # Apply pointwise convolution
        x = self.pointwise_conv(x)
        return x


# Module-level constants for shapes
IN_CHANNELS = 16
KERNEL_SIZE = 3
BATCH_SIZE = 2
INPUT_HEIGHT = 8
INPUT_WIDTH = 8

def get_inputs():
    """Return input tensors for the forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return arguments for __init__"""
    return [IN_CHANNELS, KERNEL_SIZE]