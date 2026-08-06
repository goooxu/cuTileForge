import torch
import torch.nn as nn


class Model(nn.Module):
    """GiantConv1x1ReLU (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, batch_size, height, width):
        super(Model, self).__init__()
        
        # Use 1x1 convolution for efficient computation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        # Add BatchNorm and make it deterministic
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1.eval()
        
        # Additional elementwise operations
        self.relu = nn.ReLU(inplace=True)
        
        # Store input dimensions for reference
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_size = batch_size
        self.height = height
        self.width = width
        
    def forward(self, x):
        # First convolution
        out = self.conv1(x)
        
        # Batch normalization (eval mode, deterministic)
        out = self.bn1(out)
        
        # Elementwise ReLU
        out = self.relu(out)
        
        return out


# Module-level constants for shape configuration
BATCH_SIZE = 4
IN_CHANNELS = 256
OUT_CHANNELS = 512
HEIGHT = 256
WIDTH = 256

# Pre-compute total tensor size for reference
TOTAL_ELEMENTS = BATCH_SIZE * IN_CHANNELS * HEIGHT * WIDTH


def get_inputs():
    """Generate large input tensors for benchmarking."""
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH),
    ]


def get_init_inputs():
    """Return arguments for model initialization."""
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        BATCH_SIZE,
        HEIGHT,
        WIDTH,
    ]