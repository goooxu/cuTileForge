import torch
import torch.nn as nn

"""SomeName (tier 2, pool)"""

class Model(nn.Module):
    """SomeName (tier 2, pool)"""
    
    def __init__(self, pool_size=2, out_channels=64):
        super(Model, self).__init__()
        self.pool_size = pool_size
        self.out_channels = out_channels
        
        # Conv layer to reduce channels to a manageable size
        self.conv = nn.Conv2d(3, out_channels, kernel_size=1, bias=False)
        
        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        
        # Elementwise operation (ReLU)
        self.relu = nn.ReLU(inplace=False)
        
        # BatchNorm with eval mode
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


# Module-level constants for shapes
BATCH_SIZE = 4
IN_CHANNELS = 3
IN_HEIGHT = 128
IN_WIDTH = 128
OUT_CHANNELS = 64
POOL_SIZE = 2

def get_inputs():
    # Generate a single input tensor with specified shape
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, IN_HEIGHT, IN_WIDTH)]

def get_init_inputs():
    # Return arguments for __init__ that match the constants above
    return [POOL_SIZE, OUT_CHANNELS]