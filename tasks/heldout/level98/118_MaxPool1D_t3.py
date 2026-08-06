import torch
import torch.nn as nn

"""MaxPool1D (tier 3, pool)"""

class Model(nn.Module):
    """MaxPool1D (tier 3, pool)"""

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1):
        super(Model, self).__init__()
        self.pool = nn.MaxPool1d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        # Ensure deterministic behavior by setting to eval mode
        self.pool.eval()

    def forward(self, x):
        return self.pool(x)


# Module-level constants for shapes
INPUT_BATCH_SIZE = 1
INPUT_CHANNELS = 2
INPUT_LENGTH = 16
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 0
DILATION = 1

def get_inputs():
    """Create input tensor with deterministic values."""
    x = torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    """Return arguments for __init__ method."""
    return [KERNEL_SIZE, STRIDE, PADDING, DILATION]