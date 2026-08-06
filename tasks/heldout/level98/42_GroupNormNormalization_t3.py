import torch
import torch.nn as nn

class Model(nn.Module):
    """GroupNormNormalization (tier 3, norm)"""
    def __init__(self, num_groups, num_channels, input_size):
        super(Model, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.input_size = input_size
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=1e-5)
        # Set to eval mode to ensure deterministic behavior
        self.group_norm.eval()

    def forward(self, x):
        return self.group_norm(x)

# Module-level constants for shape configuration
NUM_GROUPS = 32
NUM_CHANNELS = 256
INPUT_SIZE = (128, 256, 56, 56)  # batch=128, channels=256, height=56, width=56

def get_inputs():
    """Return list of input tensors for forward pass"""
    return [torch.randn(INPUT_SIZE, dtype=torch.float32)]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [NUM_GROUPS, NUM_CHANNELS, INPUT_SIZE]