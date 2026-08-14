import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseConv1D (tier 3, conv)"""

    def __init__(self, in_channels, kernel_size, stride, padding, dilation):
        super(Model, self).__init__()
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
IN_CHANNELS = 8
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 2
DILATION = 2
BATCH_SIZE = 4
SEQ_LEN = 32

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)]

def get_init_inputs():
    return [IN_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION]