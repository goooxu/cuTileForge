import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1.eval()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

# Shape constants
IN_CHANNELS = 3
OUT_CHANNELS = 16
BATCH_SIZE = 4
HEIGHT = 32
WIDTH = 32
KERNEL_SIZE = 3
PADDING = 1

def get_inputs():
    # Return a list with one tensor: [batch_size, channels, height, width]
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return arguments for __init__
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING]