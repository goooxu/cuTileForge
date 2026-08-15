import torch
import torch.nn as nn

class Model(nn.Module):
    """GroupNorm (tier 5, norm)"""

    def __init__(self, num_groups, num_channels, height, width, depth):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.depth = depth
        
        # Create GroupNorm layer
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        # Ensure deterministic behavior
        self.group_norm.eval()

    def forward(self, x):
        # Apply group normalization
        return self.group_norm(x)


# Module-level constants for shapes
NUM_GROUPS = 32
NUM_CHANNELS = 256
HEIGHT = 193
WIDTH = 193
DEPTH = 97
def get_inputs():
    # Return a list containing one tensor with the specified shape
    # Using contiguous memory layout for optimal performance
    return [torch.randn(DEPTH, NUM_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return arguments for __init__ in the correct order
    return [NUM_GROUPS, NUM_CHANNELS, HEIGHT, WIDTH, DEPTH]