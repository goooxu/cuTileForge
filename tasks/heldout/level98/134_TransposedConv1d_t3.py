import torch
import torch.nn as nn

"""TransposedConv1d (tier 3, conv)"""

# Module-level constants for shapes
INPUT_CHANNELS = 4
OUTPUT_CHANNELS = 6
KERNEL_SIZE = 3
STRIDE = 2
OUTPUT_PADDING = 0
BATCH_SIZE = 2
SEQUENCE_LENGTH = 8

class Model(nn.Module):
    """TransposedConv1d (tier 3, conv)"""

    def __init__(self, input_channels, output_channels, kernel_size, stride, output_padding):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose1d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=output_padding
        )

    def forward(self, x):
        return self.transposed_conv(x)


def get_inputs():
    # Input tensor with shape (batch_size, input_channels, sequence_length)
    x = torch.ones(BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH)
    return [x]


def get_init_inputs():
    # Configuration for the transposed convolution
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, STRIDE, OUTPUT_PADDING]