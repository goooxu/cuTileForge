import torch
import torch.nn as nn

class Model(nn.Module):
    """TransposedConv2d_Dilated (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups):
        super(Model, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False
        )
        # Use BatchNorm2d with eval mode for deterministic forward
        self.norm = nn.BatchNorm2d(out_channels)
        self.norm.eval()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


# Module-level constants for shapes
BATCH_SIZE = 2
IN_CHANNELS = 4
OUT_CHANNELS = 6
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
DILATION = 2
GROUPS = 1
INPUT_HEIGHT = 5
INPUT_WIDTH = 6

def get_inputs():
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION, GROUPS]