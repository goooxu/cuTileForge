import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvNormRelu (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm2d(out_channels)
        self.norm.eval()  # Make BatchNorm deterministic

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.relu(x)
        return x

# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3
BATCH_SIZE = 3
HEIGHT = 48
WIDTH = 48
def get_inputs():
    """Returns a list of tensors to pass to forward"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]