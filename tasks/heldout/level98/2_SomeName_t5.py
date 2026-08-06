import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 5, conv)"""

    def __init__(self, in_channels=64, out_channels=128, kernel_size=3, padding=1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.norm.eval()  # For deterministic behavior

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.relu(x)
        return x


# Module-level constants for shapes
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
PADDING = 1
BATCH_SIZE = 32
HEIGHT = 32
WIDTH = 32

def get_inputs():
    # Return a list with one tensor: the input to the model
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return a list of arguments to pass to __init__
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING]