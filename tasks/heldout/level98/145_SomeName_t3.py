import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""
class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        
        # Define convolution layer
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        # Define batch normalization
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()  # Make batchnorm deterministic
        
        # Additional convolution for 2-element operations
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x

INPUT_SIZE = (2, 256, 128, 128)
OUT_CHANNELS = 256
KERNEL_SIZE = 3

def get_inputs():
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    return [INPUT_SIZE[1], OUT_CHANNELS, KERNEL_SIZE]