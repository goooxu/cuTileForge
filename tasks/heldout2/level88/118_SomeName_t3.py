import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

# Module-level constants for shapes
INPUT_CHANNELS = 8
OUTPUT_CHANNELS = 8
BATCH_SIZE = 4
HEIGHT = 4
WIDTH = 4

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.norm = nn.BatchNorm2d(INPUT_CHANNELS)
        self.norm.eval()
        self.conv = nn.Conv2d(INPUT_CHANNELS, OUTPUT_CHANNELS, kernel_size=1, padding=0)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Normalization
        x = self.norm(x)
        # Residual add (x + x)
        residual = x + x
        # Convolution on the residual
        out = self.conv(residual)
        # Activation
        out = self.activation(out)
        return out

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    return []