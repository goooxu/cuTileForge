import torch
import torch.nn as nn

# Module-level constants for shape definitions
BATCH_SIZE = 2
IN_CHANNELS = 2
OUT_CHANNELS = 4
KERNEL_SIZE = 3
INPUT_HEIGHT = 12
INPUT_WIDTH = 12
class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Depthwise convolution: groups=in_channels so each input channel is convolved separately
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution (1x1 conv) to combine channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        
        # Initialize weights with deterministic values
        nn.init.kaiming_uniform_(self.depthwise_conv.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.pointwise_conv.weight, a=5**0.5)

    def forward(self, x):
        # Apply depthwise convolution
        x = self.depthwise_conv(x)
        
        # Apply pointwise convolution
        x = self.pointwise_conv(x)
        
        return x


def get_inputs():
    """Generate deterministic input tensor for the model"""
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    ]


def get_init_inputs():
    """Return arguments for model initialization"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]