import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        # Elementwise operations: ReLU and batch normalization
        self.relu = nn.ReLU(inplace=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()  # Ensure deterministic behavior
        
    def forward(self, x):
        # Convolution followed by elementwise operations
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

# Module-level constants for shapes
BATCH_SIZE = 4
IN_CHANNELS = 16
OUT_CHANNELS = 32
KERNEL_SIZE = 3
HEIGHT = 64
WIDTH = 64

def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]