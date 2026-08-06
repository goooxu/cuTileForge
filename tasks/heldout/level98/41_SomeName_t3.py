import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

class Model(nn.Module):
    """SomeName (tier 3, pool)"""

    def __init__(self, pool_kernel_size, pool_stride, pool_padding):
        super(Model, self).__init__()
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        
        # Using AvgPool2d which is deterministic
        self.pool = nn.AvgPool2d(
            kernel_size=self.pool_kernel_size,
            stride=self.pool_stride,
            padding=self.pool_padding
        )
        
    def forward(self, x):
        # Apply pooling
        pooled = self.pool(x)
        # Elementwise operation: multiply by scalar factor for variety
        # This is deterministic and uses elementwise multiplication
        return pooled * 2.0


# Module-level constants for shape configuration
BATCH_SIZE = 2
IN_CHANNELS = 4
INPUT_HEIGHT = 8
INPUT_WIDTH = 8
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
POOL_PADDING = 0

def get_inputs():
    # Return a list with one tensor: (batch_size, in_channels, height, width)
    # Using ones to make it deterministic
    return [torch.ones(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    # Return arguments for __init__ matching the constants defined above
    return [POOL_KERNEL_SIZE, POOL_STRIDE, POOL_PADDING]