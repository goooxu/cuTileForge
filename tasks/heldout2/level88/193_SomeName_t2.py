import torch
import torch.nn as nn

"""SomeName (tier 2, pool)"""


class Model(nn.Module):
    """SomeName (tier 2, pool)"""

    def __init__(self, pool_size=2):
        super(Model, self).__init__()
        self.pool_size = pool_size
        self.pool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        # No batch norm to avoid eval() requirements

    def forward(self, x):
        # Pooling layer followed by elementwise operation
        x = self.pool(x)
        # Elementwise: multiply by 2 and add 1
        x = x * 2.0 + 1.0
        return x


# Module-level constants for shape definitions
INPUT_BATCH = 2
INPUT_CHANNELS = 4
INPUT_HEIGHT = 8
INPUT_WIDTH = 8
POOL_SIZE = 2


def get_inputs():
    """Return input tensors for the model"""
    return [torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]


def get_init_inputs():
    """Return arguments for model initialization"""
    return [POOL_SIZE]