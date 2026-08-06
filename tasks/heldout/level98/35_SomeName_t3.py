import torch
import torch.nn as nn

"""
SomeName (tier 3, conv)
"""

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 16
OUT_CHANNELS = 32
KERNEL_SIZE = 3
INPUT_H = 64
INPUT_W = 64

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            stride=1,
            padding=1,
            bias=True
        )
        self.norm = nn.BatchNorm2d(num_features=OUT_CHANNELS)
        # Set to eval mode to make forward deterministic
        self.norm.eval()

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        # Elementwise operations: ReLU + BatchNorm + LeakyReLU
        x = torch.nn.functional.relu(x)
        x = self.norm(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        return x


def get_inputs():
    """Returns a list of tensors to pass to forward."""
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_H, INPUT_W)
    ]


def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return []