import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

class Model(nn.Module):
    """Depthwise Separable Convolution for medium tensors"""
    
    def __init__(self, in_channels=64, out_channels=128, kernel_size=3, bias=True):
        super(Model, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # First convolution: depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            groups=in_channels, 
            bias=bias,
            padding=kernel_size // 2
        )
        
        # Second convolution: pointwise (1x1) convolution
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=bias
        )
        
        # Set to eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # Depthwise convolution
        x = self.depthwise(x)
        # Pointwise convolution
        x = self.pointwise(x)
        return x


# Module-level constants for tensor shapes
BATCH_SIZE = 8
IN_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 32
WIDTH = 32
KERNEL_SIZE = 3


def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)]


def get_init_inputs():
    """Return initialization arguments for the model"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, True]