import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveAvgPool3D (tier 3, pool)"""

    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(output_size)

    def forward(self, x):
        return self.pool(x)

# Shape constants
BATCH_SIZE = 4
CHANNELS = 32
D_DEPTH = 10
D_HEIGHT = 12
D_WIDTH = 14

OUTPUT_SIZE = (2, 3, 4)


def get_inputs():
    # Create deterministic tensor with consistent values
    x = torch.randn(BATCH_SIZE, CHANNELS, D_DEPTH, D_HEIGHT, D_WIDTH)
    # Use consistent seed-like pattern to make it reproducible
    x = x * 0.1 + 0.5  # Scale and shift for more stable values
    return [x]


def get_init_inputs():
    return [OUTPUT_SIZE]