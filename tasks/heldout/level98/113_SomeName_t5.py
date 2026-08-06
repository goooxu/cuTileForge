import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

# Module-level constants for shapes
IN_CHANNELS = 3
OUT_CHANNELS = 6
KERNEL_SIZE = 3
INPUT_HEIGHT = 4
INPUT_WIDTH = 4
BATCH_SIZE = 2

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Depthwise convolution: groups=IN_CHANNELS means each input channel is processed separately
        self.depthwise_conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=IN_CHANNELS,
            kernel_size=KERNEL_SIZE,
            padding=KERNEL_SIZE//2,
            groups=IN_CHANNELS,
            bias=False
        )
        
        # Pointwise convolution: 1x1 convolution to combine channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=1,
            bias=False
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
    """Returns a list of tensors to pass to forward method."""
    # Create input tensor with shape (batch_size, channels, height, width)
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]


def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return []