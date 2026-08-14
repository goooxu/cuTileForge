import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.sigmoid(out)
        return out

# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 64
KERNEL_SIZE = 3
BATCH_SIZE = 1
HEIGHT = 32
WIDTH = 32

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]