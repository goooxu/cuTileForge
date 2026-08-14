import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolLayer (tier 2, pool)"""

    def __init__(self, kernel_size, stride, padding, input_channels):
        super(Model, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.input_channels = input_channels

    def forward(self, x):
        return self.pool(x)

INPUT_CHANNELS = 64
BATCH_SIZE = 16
DEPTH = 32
HEIGHT = 64
WIDTH = 64
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, DEPTH, HEIGHT, WIDTH)]

def get_init_inputs():
    return [KERNEL_SIZE, STRIDE, PADDING, INPUT_CHANNELS]