import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedDepthwiseConv1D (tier 3, conv)"""

    def __init__(self, in_channels=256, kernel_size=3, dilation=2):
        super().__init__()
        # Use depthwise convolution with dilation for high-throughput test
        self.conv = nn.Conv1d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            padding=dilation, 
            dilation=dilation, 
            groups=in_channels,  # depthwise
            bias=False
        )
        # Ensure deterministic behavior by using eval mode
        self.eval()

    def forward(self, x):
        return self.conv(x)


# Module-level constants for shapes
IN_CHANNELS = 256
KERNEL_SIZE = 3
DILATION = 2
BATCH_SIZE = 8
SEQ_LEN = 8192

def get_inputs():
    """Return list of tensors for forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [IN_CHANNELS, KERNEL_SIZE, DILATION]