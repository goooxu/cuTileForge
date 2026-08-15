import torch
import torch.nn as nn

"""LayerNorm3D (tier 3, norm)"""

# Module-level constants for shape configuration
INPUT_BATCH = 8
INPUT_SEQ_LEN = 64
INPUT_FEATURES = 257
NORM_DIM = -1

class Model(nn.Module):
    def __init__(self, normalized_shape, eps=1e-4, elementwise_affine=True):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine
        )
        self.layer_norm.eval()  # Make deterministic

    def forward(self, x):
        return self.layer_norm(x)

def get_inputs():
    # Create a tensor with shape (batch, seq_len, features) = (8, 64, 256)
    x = torch.randn(INPUT_BATCH, INPUT_SEQ_LEN, INPUT_FEATURES)
    return [x]

def get_init_inputs():
    # Initialize with the normalized shape matching the feature dimension
    return [(INPUT_FEATURES,)]