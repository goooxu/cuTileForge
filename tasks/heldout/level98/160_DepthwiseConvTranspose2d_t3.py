import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 4
IN_CHANNELS = 8
OUT_CHANNELS = 16
INPUT_SIZE = 32
KERNEL_SIZE = 3
DILATION = 2
GROUPS = 8  # depthwise convolution
STRIDE = 1
PADDING = 1
OUTPUT_PADDING = 0

class Model(nn.Module):
    """DepthwiseConvTranspose2d (tier 3, conv)"""
    
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE,
                 stride=STRIDE, padding=PADDING, dilation=DILATION, groups=GROUPS,
                 output_padding=OUTPUT_PADDING, bias=True):
        super(Model, self).__init__()
        
        # Initialize transposed depthwise convolution
        # For depthwise, out_channels must be a multiple of groups
        # and in_channels is split into groups, each processing one channel
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels * (out_channels // in_channels),  # ensuring depthwise capability
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # depthwise convolution: each input channel has its own filter
            output_padding=output_padding,
            bias=bias
        )
        
        # Set to eval mode for deterministic behavior
        self.conv_transpose.eval()
        
        # Also make sure the convolution parameters are set for proper behavior
        with torch.no_grad():
            # Initialize weights to a simple deterministic value
            nn.init.constant_(self.conv_transpose.weight, 0.1)
            if self.conv_transpose.bias is not None:
                nn.init.constant_(self.conv_transpose.bias, 0.01)

    def forward(self, x):
        # Transposed depthwise convolution operation
        # Input shape: (batch_size, in_channels, H, W)
        output = self.conv_transpose(x)
        return output  # Return single tensor


def get_inputs():
    """Returns list of tensors to pass to forward method"""
    # Create input tensor for transposed depthwise convolution
    # Shape: (batch_size, in_channels, input_size, input_size)
    input_tensor = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
    return [input_tensor]


def get_init_inputs():
    """Returns list of arguments to pass to __init__"""
    # Return parameters to reconstruct the model
    return [
        IN_CHANNELS,           # in_channels
        OUT_CHANNELS,          # out_channels
        KERNEL_SIZE,           # kernel_size
        STRIDE,                # stride
        PADDING,               # padding
        DILATION,              # dilation
        GROUPS,                # groups
        OUTPUT_PADDING,        # output_padding
        True                   # bias
    ]