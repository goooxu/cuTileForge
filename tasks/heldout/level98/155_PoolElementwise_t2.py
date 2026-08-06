import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolElementwise (tier 2, pool)"""

    def __init__(self, pool_kernel_size, pool_stride, pool_padding):
        super(Model, self).__init__()
        # Create a pooling layer (no trainable parameters, but still needs initialization)
        self.pool = nn.AvgPool2d(
            kernel_size=pool_kernel_size,
            stride=pool_stride,
            padding=pool_padding
        )
        # Add an elementwise operation that requires no training
        # Use a fixed constant for the elementwise operation to avoid randomness
        self.constant = 1.5

    def forward(self, x):
        # Pooling layer
        pooled = self.pool(x)
        # Elementwise work: multiply by constant (no randomness, no in-place modification)
        result = pooled * self.constant
        return result


# Module-level constants for shapes and configurations
INPUT_CHANNELS = 128
INPUT_HEIGHT = 128
INPUT_WIDTH = 128
BATCH_SIZE = 32
POOL_KERNEL_SIZE = 3
POOL_STRIDE = 2
POOL_PADDING = 1

def get_inputs():
    """Return input tensors for the forward pass"""
    # Create a tensor with the specified shape
    # Using torch.ones to ensure deterministic behavior
    input_tensor = torch.ones(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Return initialization arguments for the module"""
    return [POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING]