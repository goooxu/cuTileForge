import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

# Module-level constants for shapes
INPUT_CHANNELS = 64
OUTPUT_CHANNELS = 64
BATCH_SIZE = 32
FEATURE_HEIGHT = 64
FEATURE_WIDTH = 64

class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(OUTPUT_CHANNELS)
        self.norm.eval()  # Set to eval mode for deterministic behavior
        self.conv = nn.Conv2d(INPUT_CHANNELS, OUTPUT_CHANNELS, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=False)  # Non-inplace for safety
    
    def forward(self, x):
        # Store original input for residual connection
        residual = x
        
        # Apply convolution
        out = self.conv(x)
        
        # Apply normalization ( BatchNorm is deterministic in eval mode)
        out = self.norm(out)
        
        # Add residual connection
        out = out + residual
        
        # Apply activation function
        out = self.relu(out)
        
        return out


def get_inputs():
    """Generate deterministic input tensors for the model."""
    return [torch.zeros(BATCH_SIZE, INPUT_CHANNELS, FEATURE_HEIGHT, FEATURE_WIDTH)]


def get_init_inputs():
    """Return initialization arguments (empty for this model)."""
    return []