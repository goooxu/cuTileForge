import torch
import torch.nn as nn

class Model(nn.Module):
    """Pool3D (tier 5, pool)"""
    
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(Model, self).__init__()
        # Use AdaptiveAvgPool3d which is a fixed-size pooling operation
        # that doesn't require learning parameters
        self.pool = nn.AdaptiveAvgPool3d(output_size=(8, 8, 8))
    
    def forward(self, x):
        return self.pool(x)

# Module-level constants for tensor shapes
INPUT_BATCH = 2
INPUT_CHANNELS = 64
INPUT_DEPTH = 24
INPUT_HEIGHT = 24
INPUT_WIDTH = 24
OUTPUT_DEPTH = 8
OUTPUT_HEIGHT = 8
OUTPUT_WIDTH = 8

def get_inputs():
    """Return input tensor for pooling operation"""
    return [torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_DEPTH, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return []