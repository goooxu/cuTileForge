import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.BatchNorm2d(in_channels)
        self.norm.eval()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv(x)
        x = x + residual
        x = torch.relu(x)
        return x

# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 4
HEIGHT = 8
WIDTH = 8

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS]