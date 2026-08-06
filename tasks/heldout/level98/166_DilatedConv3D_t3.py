import torch
import torch.nn as nn

"""DilatedConv3D (tier 3, conv)"""

class Model(nn.Module):
    """DilatedConv3D (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              padding=dilation * (kernel_size - 1) // 2,
                              dilation=dilation, groups=groups)
        # Ensure deterministic behavior by setting model to eval mode
        self.eval()

    def forward(self, x):
        return self.conv(x)


# Module-level constants
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
DILATION = 2
GROUPS = 1
BATCH_SIZE = 4
DEPTH = 32
HEIGHT = 32
WIDTH = 32


def get_inputs():
    # Return list of tensors for forward pass
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH,
                        dtype=torch.float32, requires_grad=False)]


def get_init_inputs():
    # Return list of arguments for __init__
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION, GROUPS]