import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, elementwise)"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.sin(x)
        x = x ** 2
        x = x * 2.5
        x = torch.log1p(x)
        x = x * torch.exp(-x)
        return x

BATCH_SIZE = 32
CHANNELS = 64
HEIGHT = 64
WIDTH = 64

def get_inputs():
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return []