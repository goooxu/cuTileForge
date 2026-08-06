import torch
import torch.nn as nn

"""PoolNormConcat (pool, norm)"""


class Model(nn.Module):
    """PoolNormConcat (tier 2, pool)"""

    def __init__(self, pool_size, in_channels):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size)
        self.norm = nn.BatchNorm2d(in_channels)
        self.norm.eval()  # Make BatchNorm deterministic

    def forward(self, x):
        x = self.pool(x)
        x = self.norm(x)
        x = x * 0.5 + 0.25
        return x


INPUT_CHANNELS = 8
INPUT_HEIGHT = 16
INPUT_WIDTH = 16
POOL_SIZE = 2


def get_inputs():
    return [
        torch.randn(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
    ]


def get_init_inputs():
    return [POOL_SIZE, INPUT_CHANNELS]