import torch
import torch.nn as nn

# Module-level constants for shape definitions
BATCH_SIZE = 2
INPUT_HEIGHT = 4
INPUT_WIDTH = 4
INPUT_CHANNELS = 3
POOL_KERNEL_SIZE = 2
POOL_STRIDE = 2
OUTPUT_HEIGHT = INPUT_HEIGHT // POOL_KERNEL_SIZE
OUTPUT_WIDTH = INPUT_WIDTH // POOL_STRIDE

class Model(nn.Module):
    """AvgPool2D (tier 5, pool)"""

    def __init__(self):
        super(Model, self).__init__()
        # Use nn.AvgPool2d for average pooling
        self.pool = nn.AvgPool2d(kernel_size=POOL_KERNEL_SIZE, stride=POOL_STRIDE)
        # Set to eval mode for deterministic behavior
        self.pool.eval()

    def forward(self, x):
        # Apply average pooling without in-place modification
        return self.pool(x)


def get_inputs():
    # Create input tensor with deterministic values
    # Shape: (BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    input_tensor = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)
    return [input_tensor]


def get_init_inputs():
    # No arguments needed for __init__ in this case
    return []