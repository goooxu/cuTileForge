import torch
import torch.nn as nn

class Model(nn.Module):
    """GatingConv (tier 2, conv)"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        # Add batch norm to demonstrate eval() usage for deterministic behavior
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.eval()
        # Additional parameters for the elementwise operations
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # Convolution
        out = self.conv(x)
        # Batch normalization (already set to eval mode for deterministic behavior)
        out = self.bn(out)
        # Elementwise operation 1: ReLU
        out = torch.relu(out)
        # Elementwise operation 2: scaled addition with residual connection
        residual = x.mean(dim=1, keepdim=True) * 2
        out = out + self.alpha * residual
        # Elementwise operation 3: sigmoid scaling
        out = out * torch.sigmoid(out * self.beta)
        return out

# Module-level constants for tensor shapes
INPUT_HEIGHT = 8
INPUT_WIDTH = 8
IN_CHANNELS = 3
OUT_CHANNELS = 5
KERNEL_SIZE = 3

def get_inputs():
    """Returns a list of tensors to pass to forward()"""
    return [torch.randn(1, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__()"""
    return [IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE]