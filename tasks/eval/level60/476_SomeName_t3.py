import torch
import torch.nn as nn

"""SomeName (tier 3, conv)"""

# Module-level constants for shape configuration
BATCH_SIZE = 3
IN_CHANNELS = 4
OUT_CHANNELS = 6
KERNEL_SIZE = 3
INPUT_HEIGHT = 12
INPUT_WIDTH = 13
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        # Convolution layer
        self.conv = nn.Conv2d(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            padding=1
        )
        
        # BatchNorm layer
        self.bn = nn.BatchNorm2d(OUT_CHANNELS)
        
        # Set BatchNorm to evaluation mode for deterministic behavior
        self.bn.eval()
    
    def forward(self, x):
        # Convolution
        x = self.conv(x)
        
        # Elementwise operations: ReLU, then BatchNorm
        x = torch.relu(x)
        x = self.bn(x)
        
        return x

def get_inputs():
    """Generate input tensor for the model."""
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    ]

def get_init_inputs():
    """Return empty list since __init__ takes no arguments."""
    return []