import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvReluSigmoid (tier 2, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, height, width):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.height = height
        self.width = width
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x

# Module-level constants
IN_CHANNELS = 32
OUT_CHANNELS = 64
KERNEL_SIZE = 3
HEIGHT = 96
WIDTH = 96
def get_inputs():
    return [torch.randn(1, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, HEIGHT, WIDTH]