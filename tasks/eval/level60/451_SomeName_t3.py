import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x


# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3
BATCH_SIZE = 6
HEIGHT = 48
WIDTH = 48
def get_inputs():
    """Return input tensors for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]