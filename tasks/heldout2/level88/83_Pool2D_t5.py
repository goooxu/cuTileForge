import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 2
IN_CHANNELS = 4
HEIGHT = 8
WIDTH = 8

class Model(nn.Module):
    """Pool2D (tier 5, pool)"""
    
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        return self.pool(x)

def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, height, width)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return arguments for __init__ method
    return [2, 2, 0]