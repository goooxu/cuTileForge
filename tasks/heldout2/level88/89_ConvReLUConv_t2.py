import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvReLUConv (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.batch_norm.eval()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        return x

# Module-level constants for shapes
IN_CHANNELS = 32
OUT_CHANNELS = 64
KERNEL_SIZE = 3
BATCH_SIZE = 8
HEIGHT = 32
WIDTH = 32

def get_inputs():
    """Generate input tensors for the model."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Generate initialization arguments for the model."""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]