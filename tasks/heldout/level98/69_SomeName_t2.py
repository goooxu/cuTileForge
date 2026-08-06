import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""

class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)
        self.relu = nn.ReLU(inplace=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.batch_norm.eval()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batch_norm(x)
        x = self.sigmoid(x)
        return x

# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 6
KERNEL_SIZE = 3
INPUT_HEIGHT = 16
INPUT_WIDTH = 16

def get_inputs():
    # Returns a list containing a single input tensor for the forward pass
    return [torch.randn(1, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    # Returns arguments for __init__ method
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]