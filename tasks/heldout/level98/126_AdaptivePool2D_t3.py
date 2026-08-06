import torch
import torch.nn as nn

"""AdaptivePool2D (tier 3, pool)"""


class Model(nn.Module):
    """AdaptivePool2D (tier 3, pool)"""

    def __init__(self, output_size, input_size):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=output_size)
        # Ensure deterministic behavior
        self.pool.eval()

    def forward(self, x):
        return self.pool(x)


# Module-level constants for shapes
INPUT_SIZE = (8, 64, 128, 128)
OUTPUT_SIZE = (4, 4)


def get_inputs():
    """Return a list of tensors to pass to forward."""
    return [torch.randn(INPUT_SIZE)]


def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [OUTPUT_SIZE, INPUT_SIZE]