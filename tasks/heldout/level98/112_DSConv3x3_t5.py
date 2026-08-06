import torch
import torch.nn as nn

class Model(nn.Module):
    """DSConv3x3 (tier 5, conv)"""

    def __init__(self, in_channels=256, out_channels=256, groups=256, kernel_size=3, stride=1, padding=1):
        super(Model, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels * groups // in_channels, 
                                   kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=in_channels, bias=False)
        
        # Pointwise convolution (1x1 conv)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                   stride=1, padding=0, bias=False)
        
        # Use BatchNorm2d and set to eval mode for deterministic behavior
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()

    def forward(self, x):
        # Depthwise convolution
        out = self.depthwise(x)
        # Pointwise convolution
        out = self.pointwise(out)
        # Batch normalization
        out = self.bn(out)
        return out

# Module-level constants for tensor shapes
BATCH_SIZE = 8
IN_CHANNELS = 256
OUT_CHANNELS = 256
HEIGHT = 64
WIDTH = 64

def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [IN_CHANNELS, OUT_CHANNELS, IN_CHANNELS, 3, 1, 1]