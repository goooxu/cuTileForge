import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, pool)"""
    
    def __init__(self, kernel_size=2, stride=2):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
    
    def forward(self, x):
        x = self.pool(x)
        x = x * 2.0 + 1.0
        return x


# Module-level constants for shapes
INPUT_BATCH = 1
INPUT_CHANNELS = 4
INPUT_HEIGHT = 8
INPUT_WIDTH = 8
KERNEL_SIZE = 2
STRIDE = 2

def get_inputs():
    return [torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    return [KERNEL_SIZE, STRIDE]