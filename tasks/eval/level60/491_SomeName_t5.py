import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

# Module-level constants for shape configuration
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 64
KERNEL_SIZE = 3
INPUT_HEIGHT = 48
INPUT_WIDTH = 48
BATCH_SIZE = 2
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Depthwise convolution: groups equals input channels
        self.depthwise_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=INPUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
            groups=INPUT_CHANNELS
        )
        
        # Pointwise convolution: 1x1 convolution to mix channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=INPUT_CHANNELS,
            out_channels=OUTPUT_CHANNELS,
            kernel_size=1
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        
        # Pointwise convolution
        x = self.pointwise_conv(x)
        
        return x

def get_inputs():
    # Generate input tensor with shape (batch_size, channels, height, width)
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    # No additional initialization inputs needed
    return []