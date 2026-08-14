import torch
import torch.nn as nn

"""SomeName (tier 2, conv)"""

# Module-level constants for shapes
BATCH_SIZE = 3
IN_CHANNELS = 4
OUT_CHANNELS = 4
KERNEL_SIZE = 3
INPUT_HEIGHT = 12
INPUT_WIDTH = 12
class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups=in_channels so each input channel is processed separately
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=in_channels
        )
        
        # Pointwise convolution: 1x1 conv to combine channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )
    
    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        # Pointwise convolution
        x = self.pointwise_conv(x)
        return x

def get_inputs():
    # Create input tensor with shape (batch_size, in_channels, height, width)
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]