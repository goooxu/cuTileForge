import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

class Model(nn.Module):
    """SomeName (tier 3, pool)"""

    def __init__(self, input_channels, pool_kernel_size, pool_stride):
        super(Model, self).__init__()
        self.input_channels = input_channels
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        
        # Initialize pooling layer
        self.pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        
        # Element-wise operation parameters (learnable for some variation)
        self.scale = nn.Parameter(torch.ones(1, input_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, input_channels, 1, 1))
        
    def forward(self, x):
        # Pooling layer
        x = self.pool(x)
        
        # Element-wise work: scale and bias
        x = x * self.scale + self.bias
        
        return x


# Module-level constants for shape configuration
INPUT_BATCH = 4
INPUT_CHANNELS = 256
INPUT_HEIGHT = 768
INPUT_WIDTH = 768
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

def get_inputs():
    """Return input tensors for the model."""
    # Create input tensor with appropriate size
    x = torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]

def get_init_inputs():
    """Return arguments for model initialization."""
    return [INPUT_CHANNELS, POOL_KERNEL_SIZE, POOL_STRIDE]