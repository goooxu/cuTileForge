import torch
import torch.nn as nn

class Model(nn.Module):
    """DepthwiseConv1D (tier 2, conv)"""
    
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(Model, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Depthwise convolution: groups=in_channels, out_channels=in_channels*channel_multiplier
        # Here we use channel_multiplier=1 for simplicity
        channel_multiplier = 1
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * channel_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        
        # Set to eval mode for deterministic behavior
        self.depthwise_conv.eval()
    
    def forward(self, x):
        return self.depthwise_conv(x)


# Module-level constants for shapes
INPUT_BATCH_SIZE = 2
INPUT_CHANNELS = 3
INPUT_LENGTH = 16
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
DILATION = 1

def get_inputs():
    """Return list of input tensors for the model"""
    # Create input tensor with shape (batch_size, in_channels, length) for 1D conv
    input_tensor = torch.randn(INPUT_BATCH_SIZE, INPUT_CHANNELS, INPUT_LENGTH)
    return [input_tensor]

def get_init_inputs():
    """Return list of arguments for model initialization"""
    return [INPUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING, DILATION, True]