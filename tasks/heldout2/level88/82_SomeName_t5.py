import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Create convolution layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Use batch normalization and set to eval mode for deterministic behavior
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()

    def forward(self, x):
        # Chain of elementwise operations through convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        return x


# Module-level constants for tensor shapes
BATCH_SIZE = 8
IN_CHANNELS = 3
OUT_CHANNELS = 64
HEIGHT = 128
WIDTH = 128

def get_inputs():
    """Return list of input tensors for forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [IN_CHANNELS, OUT_CHANNELS]