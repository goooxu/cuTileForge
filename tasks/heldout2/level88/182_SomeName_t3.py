import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = torch.relu(out)
        return out

IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3

def get_inputs():
    batch_size = 4
    height = 32
    width = 32
    return [torch.randn(batch_size, IN_CHANNELS, height, width)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]