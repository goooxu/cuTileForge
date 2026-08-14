import torch
import torch.nn as nn

class Model(nn.Module):
    """GroupNorm32 (tier 3, norm)"""

    def __init__(self, num_groups, num_channels):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.gn.eval()

    def forward(self, x):
        return self.gn(x)

# Shape configuration constants
NUM_GROUPS = 32
NUM_CHANNELS = 512
BATCH_SIZE = 4
HEIGHT = 224
WIDTH = 224

def get_inputs():
    return [torch.randn(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [NUM_GROUPS, NUM_CHANNELS]