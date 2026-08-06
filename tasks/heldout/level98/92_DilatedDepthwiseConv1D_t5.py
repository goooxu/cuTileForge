import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedDepthwiseConv1D (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, groups):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        
        # Ensure deterministic behavior
        self.conv.eval()

    def forward(self, x):
        return self.conv(x)

# Shape configuration for large tensor throughput measurement
IN_CHANNELS = 1024
OUT_CHANNELS = 2048
KERNEL_SIZE = 3
DILATION = 4
GROUPS = 1024  # Depthwise convolution (groups = in_channels)

# Input tensor configuration
BATCH_SIZE = 16
SEQ_LENGTH = 4096

def get_inputs():
    """Return list of input tensors for the model."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    """Return list of arguments for __init__."""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION, GROUPS]