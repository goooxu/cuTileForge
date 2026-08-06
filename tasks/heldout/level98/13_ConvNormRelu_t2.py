import torch
import torch.nn as nn

"""ConvNormRelu (tier 2, conv)"""

class Model(nn.Module):
    """ConvNormRelu (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.norm.eval()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.relu(x)
        return x

IN_CHANNELS = 64
OUT_CHANNELS = 128
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
BATCH_SIZE = 4
HEIGHT = 56
WIDTH = 56

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]