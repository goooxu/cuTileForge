import torch
import torch.nn as nn

"""
ResidualConvBlock (tier 2, conv)
"""

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                               stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Use eval mode for batch norm to ensure deterministic behavior
        self.bn1.eval()
        self.bn2.eval()
    
    def forward(self, x):
        residual = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        out = out + residual
        out = torch.relu(out)
        
        return out


# Module-level constants for shapes
BATCH_SIZE = 1
IN_CHANNELS = 64
OUT_CHANNELS = 64
KERNEL_SIZE = 3
STRIDE = 1
PADDING = 1
INPUT_HEIGHT = 256
INPUT_WIDTH = 256

def get_inputs():
    """Return input tensors for the model"""
    # Create input tensor with appropriate shape
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]

def get_init_inputs():
    """Return arguments for model initialization"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING]