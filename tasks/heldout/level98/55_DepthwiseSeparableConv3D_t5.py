import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseSeparableConv3D (tier 5, conv)"""
    
    def __init__(self, in_channels=128, out_channels=256, kernel_size=5, padding=2, dilation=3):
        super().__init__()
        
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.batch_norm.eval()
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.batch_norm(out)
        return out


# Module-level constants for shape configuration
BATCH_SIZE = 2
IN_CHANNELS = 128
OUT_CHANNELS = 256
D = 64
H = 64
W = 64
KERNEL_SIZE = 5
PADDING = 2
DILATION = 3

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, D, H, W)]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, PADDING, DILATION]