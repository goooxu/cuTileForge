import torch
import torch.nn as nn

"""1D Transposed Convolution Model (tier 5, conv)"""

# Constants for shape configuration
INPUT_CHANNELS = 512
OUTPUT_CHANNELS = 512
KERNEL_SIZE = 3
INPUT_LENGTH = 4096
DILATION = 2
STRIDE = 2
BATCH_SIZE = 8

class Model(nn.Module):
    """1D Transposed Convolution Model (tier 5, conv)"""

    def __init__(self):
        super(Model, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(
            in_channels=INPUT_CHANNELS,
            out_channels=OUTPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=STRIDE,
            padding=1,  # ensures output size matches input * stride
            dilation=DILATION
        )
        # Set to eval mode for deterministic behavior
        self.conv1d_transpose.eval()

    def forward(self, x):
        return self.conv1d_transpose(x)

def get_inputs():
    """Return input tensor for forward pass"""
    return [
        torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH, dtype=torch.float32)
    ]

def get_init_inputs():
    """Return empty list since __init__ takes no arguments"""
    return []