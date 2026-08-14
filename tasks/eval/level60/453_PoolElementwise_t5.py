import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolElementwise (tier 5, pool)"""

    def __init__(self, pool_kernel_size, pool_stride, pool_padding, add_value, mul_value):
        super(Model, self).__init__()
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.add_value = add_value
        self.mul_value = mul_value
        self.avg_pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
    
    def forward(self, x):
        pooled = self.avg_pool(x)
        result = pooled + self.add_value
        result = result * self.mul_value
        return result

# Module-level constants for shape configuration
BATCH_SIZE = 6
CHANNELS = 128
HEIGHT = 96
WIDTH = 96
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
POOL_PADDING = 0
ADD_VALUE = 1.5
MUL_VALUE = 2.0

def get_inputs():
    """Return list of input tensors for the model"""
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return list of arguments for model initialization"""
    return [POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING, ADD_VALUE, MUL_VALUE]