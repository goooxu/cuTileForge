import torch
import torch.nn as nn

class Model(nn.Module):
    """ConvReluSigmoid (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, height, width):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.height = height
        self.width = width
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv.eval()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x


INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 8
KERNEL_SIZE = 3
BATCH_SIZE = 2
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)]

def get_init_inputs():
    return [INPUT_CHANNELS, OUTPUT_CHANNELS, KERNEL_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH]