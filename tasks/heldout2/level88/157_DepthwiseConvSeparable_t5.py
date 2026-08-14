import torch
import torch.nn as nn

INPUT_CHANNELS = 4
DEPTH_MULTIPLIER = 1
KERNEL_SIZE = 3
INPUT_HEIGHT = 8
INPUT_WIDTH = 8

class Model(nn.Module):
    """DepthwiseConvSeparable (tier 5, conv)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
        self.depthwise_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=INPUT_CHANNELS * DEPTH_MULTIPLIER,
            kernel_size=KERNEL_SIZE,
            groups=INPUT_CHANNELS,
            bias=False
        )
        
        self.pointwise_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS * DEPTH_MULTIPLIER,
            out_channels=INPUT_CHANNELS * DEPTH_MULTIPLIER,
            kernel_size=1,
            bias=False
        )
    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

def get_inputs():
    batch_size = 2
    return [torch.randn(batch_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    return []