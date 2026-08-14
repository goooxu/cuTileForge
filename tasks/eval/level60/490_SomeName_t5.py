import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.relu(out)
        out = torch.tanh(out)
        return out


# Module-level constants for shape configuration
IN_CHANNELS = 32
OUT_CHANNELS = 64
KERNEL_SIZE = 3
BATCH_SIZE = 6
HEIGHT = 96
WIDTH = 96
def get_inputs():
    """Returns list of input tensors for the model's forward pass"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    """Returns list of arguments for model initialization"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]