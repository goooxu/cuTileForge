import torch
import torch.nn as nn

class Model(nn.Module):
    """DilatedDepthwiseConv1D (tier 5, conv)"""
    
    def __init__(self, in_channels, kernel_size, dilation, groups, stride):
        super(Model, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        # Set to eval mode for deterministic behavior
        self.conv.eval()
    
    def forward(self, x):
        return self.conv(x)

# Module-level constants for tensor shapes
IN_CHANNELS = 256
KERNEL_SIZE = 3
DILATION = 4
GROUPS = 256
STRIDE = 1
BATCH_SIZE = 48
SEQ_LEN = 1024

def get_inputs():
    """Generate input tensor for the dilated depthwise convolution"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)]

def get_init_inputs():
    """Generate initialization arguments for the model"""
    return [IN_CHANNELS, KERNEL_SIZE, DILATION, GROUPS, STRIDE]