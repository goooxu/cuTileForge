import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNorm (tier 2, norm)"""

    def __init__(self, normalized_shape=(64, 64), eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # Make it eval mode for deterministic behavior
        self.eval()

    def forward(self, x):
        # RMS normalization: normalize by root mean square of the values
        # along the last normalized_shape dimensions
        # x shape: (batch, *normalized_shape)
        # Compute RMS over the last len(normalized_shape) dimensions
        input_shape = x.shape
        # Reshape to (batch, *normalized_shape)
        batch_size = input_shape[0]
        # Normalize over the last len(self.normalized_shape) dimensions
        x_rms = torch.rsqrt((x ** 2).mean(dim=-len(self.normalized_shape), keepdim=True) + self.eps)
        normalized = x * x_rms
        # Apply weight
        output = normalized * self.weight
        return output


# Module-level constants for shapes
BATCH_SIZE = 4
NORMALIZED_SHAPE = (64, 64)
INPUT_SHAPE = (BATCH_SIZE, *NORMALIZED_SHAPE)

def get_inputs():
    """Returns a list of input tensors for forward pass"""
    return [torch.randn(INPUT_SHAPE)]

def get_init_inputs():
    """Returns a list of arguments for model initialization"""
    return [NORMALIZED_SHAPE, 1e-5]