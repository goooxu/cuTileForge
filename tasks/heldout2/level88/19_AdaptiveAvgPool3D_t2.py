import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveAvgPool3D (tier 2, pool)"""
    
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, x):
        return nn.functional.adaptive_avg_pool3d(x, self.output_size)


# Module-level constants for shapes
BATCH_SIZE = 4
IN_CHANNELS = 128
DEPTH = 64
HEIGHT = 64
WIDTH = 64
OUTPUT_SIZE = (4, 4, 4)

def get_inputs():
    """Generate input tensor for the model"""
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [OUTPUT_SIZE]