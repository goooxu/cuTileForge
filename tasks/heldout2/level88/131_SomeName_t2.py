import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, pool)"""

    def __init__(self, pool_kernel_size=2, pool_stride=2, input_channels=64, input_height=512, input_width=512):
        super().__init__()
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        # Create a learnable weight tensor for elementwise multiplication
        # Using nn.Parameter so it's tracked by the model
        self.weight = nn.Parameter(torch.ones(1, input_channels, input_height // pool_stride, input_width // pool_stride))
        
        # Initialize weight with ones for deterministic behavior
        with torch.no_grad():
            self.weight.fill_(1.0)

    def forward(self, x):
        # Apply max pooling
        pooled = nn.functional.max_pool2d(x, kernel_size=self.pool_kernel_size, stride=self.pool_stride)
        
        # Elementwise multiplication with learnable weight
        result = pooled * self.weight
        
        return result

# Module-level constants for tensor shapes
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
INPUT_CHANNELS = 64
INPUT_HEIGHT = 512
INPUT_WIDTH = 512
BATCH_SIZE = 8

def get_inputs():
    """Returns a list of tensors to pass to forward"""
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__"""
    return [POOL_KERNEL_SIZE, POOL_STRIDE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH]