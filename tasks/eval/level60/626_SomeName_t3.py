import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

# Module-level constants for shapes
BATCH_SIZE = 12
IN_CHANNELS = 32
OUT_CHANNELS = 32
INPUT_HEIGHT = 96
INPUT_WIDTH = 96
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

class Model(nn.Module):
    """SomeName (tier 3, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Conv layer to transform input channels
        self.conv = nn.Conv2d(IN_CHANNELS, OUT_CHANNELS, kernel_size=KERNEL_SIZE, 
                              stride=STRIDE, padding=PADDING)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        # Batch normalization
        self.bn = nn.BatchNorm2d(OUT_CHANNELS)
        # Set to eval mode for deterministic behavior
        self.bn.eval()
    
    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Apply ReLU activation
        x = torch.relu(x)
        # Apply pooling
        x = self.pool(x)
        # Apply batch normalization
        x = self.bn(x)
        # Element-wise operation (ReLU)
        x = torch.relu(x)
        return x

def get_inputs():
    """Return input tensors for the model"""
    # Create input tensor with shape (batch_size, in_channels, height, width)
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return []