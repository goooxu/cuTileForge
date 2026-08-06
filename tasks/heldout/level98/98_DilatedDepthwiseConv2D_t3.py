import torch
import torch.nn as nn

"""
DilatedDepthwiseConv2D (tier 3, conv)
"""

# Module-level constants
N = 2  # Batch size
C = 32  # Number of input channels (matches output channels for depthwise)
H = 64  # Input height
W = 64  # Input width
K = 3  # Kernel size
D = 2  # Dilation rate
S = 1  # Stride
P = 2  # Padding

class Model(nn.Module):
    """DilatedDepthwiseConv2D (tier 3, conv)"""

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=C,
            out_channels=C,
            kernel_size=K,
            stride=S,
            padding=P,
            dilation=D,
            groups=C,  # Depthwise convolution: groups = in_channels
            bias=False
        )
        # Set to eval mode to ensure deterministic behavior
        self.conv.eval()

    def forward(self, x):
        return self.conv(x)

def get_inputs():
    """Returns a list of input tensors for the forward pass."""
    return [torch.randn(N, C, H, W)]

def get_init_inputs():
    """Returns a list of arguments for __init__ (empty for this module)."""
    return []