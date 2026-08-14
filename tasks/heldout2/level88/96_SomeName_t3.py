import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, pool)"""
    
    def __init__(self, pool_kernel_size, pool_stride):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        
    def forward(self, x):
        # Apply max pooling
        pooled = self.pool(x)
        # Elementwise work: apply tanh activation
        result = torch.tanh(pooled)
        return result

# Module-level constants for shapes
INPUT_HEIGHT = 16
INPUT_WIDTH = 16
INPUT_CHANNELS = 3
BATCH_SIZE = 4
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2

def get_inputs():
    """Generate input tensors for the model."""
    # Create a sample input tensor with shape (batch_size, channels, height, width)
    x = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]

def get_init_inputs():
    """Return arguments to pass to __init__."""
    return [POOL_KERNEL_SIZE, POOL_STRIDE]