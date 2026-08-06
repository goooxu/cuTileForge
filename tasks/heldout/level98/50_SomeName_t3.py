import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

class Model(nn.Module):
    def __init__(self, pool_size, pool_stride):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        
    def forward(self, x):
        x = self.pool(x)
        x = x * 0.5  # Scale operation after pooling
        x = x + 0.1  # Bias operation after pooling
        return x

# Module-level constants for shape configuration
BATCH_SIZE = 64
CHANNELS = 256
HEIGHT = 112
WIDTH = 112
POOL_SIZE = 2
POOL_STRIDE = 2

def get_inputs():
    return [
        torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    ]

def get_init_inputs():
    return [POOL_SIZE, POOL_STRIDE]