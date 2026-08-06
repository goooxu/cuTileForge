import torch
import torch.nn as nn

INPUT_H = 5
INPUT_W = 7
OUTPUT_CHANNELS = 3
KERNEL_SIZE = 3
DILATION = 2
STRIDE = 1

class Model(nn.Module):
    """DilatedConv2d (tier 2, conv)"""
    
    def __init__(self, input_h, input_w, output_channels, kernel_size, dilation, stride):
        super(Model, self).__init__()
        self.input_h = input_h
        self.input_w = input_w
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        
        # Calculate input channels based on input dimensions
        self.in_channels = 1
        
        # Create a 2D dilated convolution
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=(dilation * (kernel_size - 1)) // 2  # Maintain spatial size
        )
        
        # Set to eval mode for deterministic behavior
        self.conv.eval()
    
    def forward(self, x):
        return self.conv(x)

def get_inputs():
    # Create input tensor with the specified dimensions
    x = torch.randn(1, 1, INPUT_H, INPUT_W)
    return [x]

def get_init_inputs():
    return [INPUT_H, INPUT_W, OUTPUT_CHANNELS, KERNEL_SIZE, DILATION, STRIDE]