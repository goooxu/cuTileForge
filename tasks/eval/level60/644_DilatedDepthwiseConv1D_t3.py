import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedDepthwiseConv1D (tier 3, conv)"""

    def __init__(self, in_channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Depthwise convolution with dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,  # depthwise
            dilation=dilation,
            bias=False
        )
        
        # Initialize weights to be deterministic
        nn.init.constant_(self.conv.weight, 0.5)

    def forward(self, x):
        return self.conv(x)

# Module-level constants for shape configuration
IN_CHANNELS = 4
KERNEL_SIZE = 3
DILATION = 2
BATCH_SIZE = 3
SEQ_LENGTH = 10

def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [IN_CHANNELS, KERNEL_SIZE, DILATION]