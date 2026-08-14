import torch
import torch.nn as nn

"""LayerNormExample (tier 5, norm)"""

# Module-level constants for shape configuration
BATCH_SIZE = 2
SEQ_LEN = 4
FEATURE_DIM = 8
NORM_DIM = -1

class Model(nn.Module):
    """LayerNormExample (tier 5, norm)"""

    def __init__(self, feature_dim, eps=1e-5):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(feature_dim, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)

def get_inputs():
    # Create a tensor with shape [BATCH_SIZE, SEQ_LEN, FEATURE_DIM]
    # This represents a typical use case for LayerNorm
    x = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM)
    return [x]

def get_init_inputs():
    # Return the arguments for __init__ - feature_dim matches the last dimension
    return [FEATURE_DIM]