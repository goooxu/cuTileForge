import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptiveMaxPool3D (tier 3, pool)"""

    def __init__(self, output_size, input_shape):
        super(Model, self).__init__()
        self.output_size = output_size
        self.input_shape = input_shape
        self.pool = nn.AdaptiveMaxPool3d(output_size)

    def forward(self, x):
        return self.pool(x)

# Module-level constants for shape configuration
OUTPUT_SIZE = (4, 4, 4)
INPUT_SHAPE = (1, 32, 128, 128, 128)

def get_inputs():
    """Return input tensor for the model."""
    return [torch.randn(INPUT_SHAPE)]

def get_init_inputs():
    """Return arguments for model initialization."""
    return [OUTPUT_SIZE, INPUT_SHAPE]
_EVAL_MARK = 1
