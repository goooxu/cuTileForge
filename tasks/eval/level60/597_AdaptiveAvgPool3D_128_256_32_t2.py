import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveAvgPool3D_128_256_32 (tier 2, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
    def forward(self, x):
        return self.pool(x)

# Module-level constants for shapes
INPUT_BATCH = 2
INPUT_CHANNELS = 128
INPUT_D = 32
INPUT_H = 32
INPUT_W = 32

def get_inputs():
    # Create a large tensor suitable for measuring throughput
    return [torch.randn(INPUT_BATCH, INPUT_CHANNELS, INPUT_D, INPUT_H, INPUT_W)]

def get_init_inputs():
    # No additional inputs needed for initialization
    return []
_EVAL_MARK = 1
