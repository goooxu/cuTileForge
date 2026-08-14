import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Model, self).__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        
        # Elementwise operations: ReLU and BatchNorm
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # Set batch norm to eval mode for deterministic behavior
        self.bn.eval()

    def forward(self, x):
        # Convolution
        x = self.conv(x)
        
        # Elementwise operations: ReLU then BatchNorm
        x = self.relu(x)
        x = self.bn(x)
        
        return x

# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 64
KERNEL_SIZE = 3
PADDING = 1
BATCH_SIZE = 6
HEIGHT = 48
WIDTH = 48
def get_inputs():
    """Return a list of tensors to pass to forward."""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING]