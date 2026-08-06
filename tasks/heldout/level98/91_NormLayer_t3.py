import torch
import torch.nn as nn

"""NormLayer (tier 3, norm)"""

INPUT_CHANNELS = 128
INPUT_HEIGHT = 64
INPUT_WIDTH = 64
BATCH_SIZE = 8

class Model(nn.Module):
    """NormLayer (tier 3, norm)"""
    
    def __init__(self, num_groups=32):
        super(Model, self).__init__()
        self.num_groups = num_groups
        # Use GroupNorm with num_groups groups and num_channels channels
        self.group_norm = nn.GroupNorm(num_groups, INPUT_CHANNELS)
        
    def forward(self, x):
        # GroupNorm expects shape (N, C, *) where C = INPUT_CHANNELS
        return self.group_norm(x)

def get_inputs():
    # Create a tensor of shape (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    return [32]  # num_groups=32 matches the INPUT_CHANNELS division (128/32=4 channels per group)