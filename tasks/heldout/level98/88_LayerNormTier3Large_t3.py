import torch
import torch.nn as nn

"""LayerNormTier3Large (tier 3, norm)"""

NUM_BATCHES = 2
NUM_CHANNELS = 512
HEIGHT = 512
WIDTH = 512
EPSILON = 1e-5

class Model(nn.Module):
    def __init__(self, normalized_shape, eps=EPSILON):
        super(Model, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps=eps)
        # Ensure deterministic behavior by setting eval mode
        self.layernorm.eval()

    def forward(self, x):
        return self.layernorm(x)

def get_inputs():
    """Returns a list containing one large tensor for LayerNorm."""
    # Create a large tensor suitable for performance measurement
    # Shape: (num_batches, num_channels, height, width)
    return [torch.randn(NUM_BATCHES, NUM_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Returns the arguments needed for __init__."""
    # For LayerNorm, we need to specify the normalized_shape
    # Since we're normalizing over the last dimension (width), we normalize over WIDTH
    return [WIDTH]