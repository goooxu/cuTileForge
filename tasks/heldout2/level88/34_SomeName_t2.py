import torch
import torch.nn as nn

"""SomeName (tier 2, pool)"""

# Module-level constants for shapes
INPUT_CHANNELS = 64
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
OUTPUT_CHANNELS = 64
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

class Model(nn.Module):
    """SomeName (tier 2, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Using AdaptiveAvgPool2d to ensure output is consistent regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((64, 64))
        
    def forward(self, x):
        # Apply pooling layer
        x = self.pool(x)
        # Apply elementwise operation (ReLU)
        x = torch.relu(x)
        return x

def get_inputs():
    # Generate medium-sized tensor for pooling operation
    batch_size = 4
    channels = INPUT_CHANNELS
    height = INPUT_HEIGHT
    width = INPUT_WIDTH
    return [torch.randn(batch_size, channels, height, width)]

def get_init_inputs():
    # No additional inputs needed for initialization
    return []