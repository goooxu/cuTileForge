import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        
        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Elementwise operations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Set to eval mode for deterministic behavior
        self.conv.eval()
        
    def forward(self, x):
        # Convolution followed by elementwise operations
        out = self.conv(x)
        out = self.relu(out)
        out = self.sigmoid(out)
        return out

# Module-level constants for shapes
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
BATCH_SIZE = 4
HEIGHT = 64
WIDTH = 64

def get_inputs():
    """Returns a list of tensors to pass to forward"""
    # Create input tensor with shape (batch_size, in_channels, height, width)
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    return [x]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]