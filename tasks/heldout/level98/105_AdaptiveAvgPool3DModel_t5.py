import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveAvgPool3DModel (tier 5, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        # Adaptive average pooling layer - no learnable parameters, just configuration
        self.pool = nn.AdaptiveAvgPool3d(output_size=(4, 4, 4))
    
    def forward(self, x):
        return self.pool(x)

# Module-level constants for shapes
INPUT_CHANNELS = 128
INPUT_D = 64
INPUT_H = 64
INPUT_W = 64
OUTPUT_D = 4
OUTPUT_H = 4
OUTPUT_W = 4

def get_inputs():
    """Return input tensor for forward pass"""
    return [torch.randn(INPUT_CHANNELS, INPUT_D, INPUT_H, INPUT_W, dtype=torch.float32)]

def get_init_inputs():
    """Return arguments for model initialization"""
    return []