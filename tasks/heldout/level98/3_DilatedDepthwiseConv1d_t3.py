import torch
import torch.nn as nn

"""DilatedDepthwiseConv1d (tier 3, conv)"""

INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 64
KERNEL_SIZE = 5
DILATION = 3
STRIDE = 1
PADDING = (KERNEL_SIZE - 1) * DILATION // 2
BATCH_SIZE = 4
SEQ_LENGTH = 128

class Model(nn.Module):
    """DilatedDepthwiseConv1d (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        
        # Use depthwise convolution (groups = in_channels)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        
        # Set model to eval mode for deterministic behavior
        self.eval()
    
    def forward(self, x):
        return self.conv(x)


def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, seq_length)
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQ_LENGTH, requires_grad=False)]

def get_init_inputs():
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, DILATION, STRIDE, PADDING]