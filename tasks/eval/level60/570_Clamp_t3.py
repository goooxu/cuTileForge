import torch
import torch.nn as nn

class Model(nn.Module):
    """Clamp (tier 3, norm)"""

    def __init__(self, num_groups, num_channels):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.gn.eval()

    def forward(self, x):
        return torch.clamp(self.gn(x), min=-1.0, max=1.0)

# Shape configuration constants
NUM_GROUPS = 32
NUM_CHANNELS = 512
BATCH_SIZE = 6
HEIGHT = 336
WIDTH = 336
def get_inputs():
    return [torch.randn(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [NUM_GROUPS, NUM_CHANNELS]