import torch
import torch.nn as nn

"""
DepthwiseTransposedConv2d (tier 5, conv)
"""

# Shape constants
BATCH_SIZE = 2
IN_CHANNELS = 16
INPUT_HEIGHT = 4
INPUT_WIDTH = 4
KERNEL_SIZE = 3
OUTPUT_PADDING = 1
STRIDE = 2
DILATION = 1
GROUPS = 16

# Output channels should equal in_channels for depthwise
OUT_CHANNELS = IN_CHANNELS

class Model(nn.Module):
    """DepthwiseTransposedConv2d (tier 5, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0, groups=1, dilation=1):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
            bias=False
        )
        # Set to eval mode for deterministic behavior
        self.conv.eval()

    def forward(self, x):
        return self.conv(x)

def get_inputs():
    """Return input tensors for the forward pass"""
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]

def get_init_inputs():
    """Return arguments for __init__"""
    return [
        IN_CHANNELS,
        OUT_CHANNELS,
        KERNEL_SIZE,
        STRIDE,
        0,
        OUTPUT_PADDING,
        GROUPS,
        DILATION
    ]