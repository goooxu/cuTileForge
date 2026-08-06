import torch
import torch.nn as nn

"""SomeName (tier 2, pool)"""

class Model(nn.Module):
    """SomeName (tier 2, pool)"""
    def __init__(self, pool_size, pool_stride, padding=0):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_stride, padding=padding)
        self.bn = nn.BatchNorm2d(4)
        self.bn.eval()
    
    def forward(self, x):
        # Pooling layer
        pooled = self.pool(x)
        # BatchNorm (evaluated so deterministic)
        normalized = self.bn(pooled)
        # Elementwise multiplication by a scalar
        result = normalized * 1.5
        return result

# Module-level constants for shapes
INPUT_BATCH = 2
INPUT_CHANNELS = 4
INPUT_HEIGHT = 8
INPUT_WIDTH = 8
POOL_SIZE = 2
POOL_STRIDE = 2

def get_inputs():
    # Create input tensor
    x = torch.ones(INPUT_BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]

def get_init_inputs():
    return [POOL_SIZE, POOL_STRIDE]