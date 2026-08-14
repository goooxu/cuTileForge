import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, pool)"""

    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.scale = nn.Parameter(torch.tensor(2.0))
    
    def forward(self, x):
        x = self.pool(x)
        x = x * self.scale
        return x

INPUT_HEIGHT = 8
INPUT_WIDTH = 8
INPUT_CHANNELS = 4
BATCH_SIZE = 2

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    return []