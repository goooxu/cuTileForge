import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseConv1D (tier 3, conv)"""
    
    def __init__(self, in_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels
        )
    
    def forward(self, x):
        return self.conv(x)


# Module-level constants for shapes
IN_CHANNELS = 4
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DILATION = 1
BATCH_SIZE = 2
SEQ_LENGTH = 8

def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, seq_length)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LENGTH)]

def get_init_inputs():
    return [IN_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION]