import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolConv (tier 3, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)  # Using False to avoid in-place modification
        
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


# Module-level constants for shape configuration
IN_CHANNELS = 256
OUT_CHANNELS = 512
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
BATCH_SIZE = 16
HEIGHT = 224
WIDTH = 224

def get_inputs():
    # Return a list with a single tensor for the model input
    # Size: (batch_size, in_channels, height, width)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return arguments for __init__ method
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]