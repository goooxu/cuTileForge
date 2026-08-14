import torch
import torch.nn as nn

"""Conv1D (tier 3, conv)"""

# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 4
OUT_CHANNELS = 6
KERNEL_SIZE = 3
INPUT_LENGTH = 8
DILATION = 2
PADDING = 1
STRIDE = 1

class Model(nn.Module):
    """Conv1D (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, input_length, dilation, padding, stride):
        super(Model, self).__init__()
        
        # Create a 1D transposed convolution layer
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        
    def forward(self, x):
        return self.conv1d_transpose(x)

def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, input_length)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_LENGTH)]

def get_init_inputs():
    # Return the parameters needed for __init__
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, INPUT_LENGTH, DILATION, PADDING, STRIDE]