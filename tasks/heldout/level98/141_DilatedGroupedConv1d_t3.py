import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedGroupedConv1d (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, groups, kernel_size, dilation, bias=True):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        self.conv.eval()  # For deterministic behavior

    def forward(self, x):
        return self.conv(x)


# Module-level constants for shapes
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
GROUPS = 8
KERNEL_SIZE = 3
DILATION = 2
BATCH_SIZE = 4
SEQUENCE_LENGTH = 256
DROPOUT_P = 0.1

def get_inputs():
    return [
        torch.randn(BATCH_SIZE, INPUT_CHANNELS, SEQUENCE_LENGTH)
    ]

def get_init_inputs():
    return [
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
        GROUPS,
        KERNEL_SIZE,
        DILATION
    ]