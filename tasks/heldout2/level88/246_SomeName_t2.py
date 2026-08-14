import torch
import torch.nn as nn

# Module-level constants for shape configuration
IN_CHANNELS = 3
OUT_CHANNELS = 12
KERNEL_SIZE = 3
INPUT_HEIGHT = 8
INPUT_WIDTH = 8
BATCH_SIZE = 2

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
        # Depthwise convolution: each input channel processed separately
        self.depthwise_conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
            groups=IN_CHANNELS
        )
        
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=1
        )
    
    def forward(self, x):
        # Apply depthwise convolution
        x = self.depthwise_conv(x)
        
        # Apply pointwise convolution
        x = self.pointwise_conv(x)
        
        return x


def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]


def get_init_inputs():
    """Return initialization arguments (empty since __init__ takes no args)"""
    return []