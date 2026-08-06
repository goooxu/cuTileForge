import torch
import torch.nn as nn

"""SomeName (tier 3, pool)"""

INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_CHANNELS = 32
POOL_KERNEL_SIZE = 2
POOL_PADDING = 0
POOL_STRIDE = 2
OUTPUT_HEIGHT = INPUT_HEIGHT // POOL_KERNEL_SIZE
OUTPUT_WIDTH = INPUT_WIDTH // POOL_KERNEL_SIZE


class Model(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, pool_kernel_size=POOL_KERNEL_SIZE,
                 pool_padding=POOL_PADDING, pool_stride=POOL_STRIDE):
        super(Model, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, padding=pool_padding, stride=pool_stride)
        self.register_buffer('scale', torch.tensor(1.5))
        self.register_buffer('bias', torch.tensor(0.1))
        self._is_evaluated = False

    def forward(self, x):
        pooled = self.pool(x)
        scaled = pooled * self.scale
        result = scaled + self.bias
        return result


def get_inputs():
    input_tensor = torch.randn(INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]


def get_init_inputs():
    return [INPUT_CHANNELS, POOL_KERNEL_SIZE, POOL_PADDING, POOL_STRIDE]