import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, pool)"""
    
    def __init__(self, pool_kernel_size=2, pool_stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        
    def forward(self, x):
        x = self.pool(x)
        x = x * 2.0
        return x

# Module-level constants for shapes
INPUT_HEIGHT = 8
INPUT_WIDTH = 8
INPUT_CHANNELS = 4
BATCH_SIZE = 2
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    return [POOL_KERNEL_SIZE, POOL_STRIDE]