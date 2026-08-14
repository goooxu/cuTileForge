import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedConv1D (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride=1):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=True
        )
        
    def forward(self, x):
        return self.conv(x)

# Module-level constants
BATCH_SIZE = 4
IN_CHANNELS = 16
OUT_CHANNELS = 32
SEQ_LENGTH = 128
KERNEL_SIZE = 3
DILATION = 2
STRIDE = 1

def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, seq_length)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    # Return arguments for __init__
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION, STRIDE]