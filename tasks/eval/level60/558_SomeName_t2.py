import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, pool)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv1x1(x)
        x = self.relu(x)
        return x

INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 128
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
BATCH_SIZE = 6
HEIGHT = 84
WIDTH = 84
def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]