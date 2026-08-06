import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedDepthwiseConv1D (tier 5, conv)"""

    def __init__(self, in_channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Depthwise transposed 1D convolution with dilation
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            groups=in_channels,
            dilation=dilation,
            bias=False
        )
        
        # Set to eval mode for deterministic behavior
        self.conv.eval()

    def forward(self, x):
        return self.conv(x)


# Module-level constants for shape configuration
IN_CHANNELS = 1024
KERNEL_SIZE = 5
DILATION = 3
BATCH_SIZE = 8
SEQUENCE_LENGTH = 2048

def get_inputs():
    """Return input tensors for forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQUENCE_LENGTH)]

def get_init_inputs():
    """Return arguments for __init__"""
    return [IN_CHANNELS, KERNEL_SIZE, DILATION]