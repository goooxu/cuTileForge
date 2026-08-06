import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

# Module-level constants for shape configuration
BATCH_SIZE = 2
IN_CHANNELS = 4
OUT_CHANNELS = 8
INPUT_HEIGHT = 5
INPUT_WIDTH = 5
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Depthwise convolution: groups = in_channels
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise convolution: 1x1 convolution to mix channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # For deterministic behavior in eval mode
        self.eval()

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise_conv(x)
        # Pointwise convolution
        x = self.pointwise_conv(x)
        return x

def get_inputs():
    """Generate input tensors for the model."""
    # Create input tensor with shape (batch_size, in_channels, height, width)
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Return initialization arguments for the model."""
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        PADDING
    ]