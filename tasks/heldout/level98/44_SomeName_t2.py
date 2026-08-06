import torch
import torch.nn as nn

"""
ConvReLU (tier 2, conv)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 56
WIDTH = 56
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, 
                 kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING):
        super(Model, self).__init__()
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.batch_norm.eval()  # Make deterministic
        
        # Elementwise operations (ReLU + multiplication by 1.1)
        self.relu = nn.ReLU()
        self.scale_factor = nn.Parameter(torch.tensor(1.1))

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x * self.scale_factor
        return x


def get_inputs():
    """Return list of input tensors for forward pass."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    """Return list of arguments for model initialization."""
    return []