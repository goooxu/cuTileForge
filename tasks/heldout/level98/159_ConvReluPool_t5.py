import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvReluPool (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.pool = nn.MaxPool2d(pool_size, stride=pool_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        return x

INPUT_HEIGHT = 16
INPUT_WIDTH = 16
IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3
POOL_SIZE = 2
BATCH_SIZE = 4

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, POOL_SIZE]