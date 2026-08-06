import torch
import torch.nn as nn

"""DepthwiseConv2d (tier 2, conv)"""

# Module-level constants for shape configuration
BATCH_SIZE = 16
IN_CHANNELS = 256
OUT_CHANNELS = 256
KERNEL_SIZE = 3
INPUT_HEIGHT = 128
INPUT_WIDTH = 128

class Model(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
                 kernel_size=KERNEL_SIZE, input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH):
        super(Model, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.input_height = input_height
        self.input_width = input_width
        
        # Depthwise convolution: groups=in_channels, each input channel is convolved independently
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            groups=in_channels,  # depthwise convolution
            padding=kernel_size//2
        )
        
        # Pointwise convolution (1x1) to combine channels
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            groups=1
        )
        
        # Use BatchNorm with eval() mode for deterministic forward pass
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.bn.eval()
    
    def forward(self, x):
        # First: depthwise convolution
        out = self.depthwise_conv(x)
        
        # Second: pointwise convolution
        out = self.pointwise_conv(out)
        
        # Optional: apply batch normalization
        out = self.bn(out)
        
        return out

def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return []