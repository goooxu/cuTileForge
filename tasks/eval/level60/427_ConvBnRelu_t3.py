import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvBnRelu (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, height, width):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# Module-level constants for shape parameters
IN_CHANNELS = 256
OUT_CHANNELS = 512
KERNEL_SIZE = 3
BATCH_SIZE = 24
HEIGHT = 384
WIDTH = 384
def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, HEIGHT, WIDTH]