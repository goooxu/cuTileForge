import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptivePool2D (tier 2, pool)"""

    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
        self.pool.eval()

    def forward(self, x):
        return self.pool(x)

INPUT_CHANNELS = 128
OUTPUT_CHANNELS = 128
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
OUTPUT_SIZE = 7

def get_inputs():
    return [torch.randn(1, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    return [OUTPUT_SIZE]