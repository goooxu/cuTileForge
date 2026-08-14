import torch
import torch.nn as nn

class Model(nn.Module):
    """HardSwish (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=dilation * (kernel_size - 1) // 2,
                             dilation=dilation, bias=True)

    def forward(self, x):
        return torch.nn.functional.hardswish(self.conv(x))

# Module-level constants
IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
DILATION = 2
BATCH_SIZE = 12
HEIGHT = 84
WIDTH = 84
def get_inputs():
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH, dtype=torch.float32)
    return [x]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATION]