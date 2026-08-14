import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""


class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Model, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.norm.eval()  # Ensure deterministic behavior
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


# Module-level constants for shapes
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 64
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
BATCH_SIZE = 12
HEIGHT = 48
WIDTH = 48
def get_inputs():
    """Return input tensors for forward pass"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    """Return arguments for __init__"""
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]