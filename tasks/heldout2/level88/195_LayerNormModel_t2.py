import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormModel (tier 2, norm)"""

    def __init__(self, normalized_shape, eps=1e-5):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.layer_norm.eval()  # Set to eval mode for deterministic behavior

    def forward(self, x):
        return self.layer_norm(x)

# Module-level constants for shapes
BATCH_SIZE = 4
SEQ_LEN = 128
HIDDEN_DIM = 512

def get_inputs():
    # Generate a tensor with shape [batch_size, seq_len, hidden_dim]
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)]

def get_init_inputs():
    # Return the normalized shape for LayerNorm (last dimension)
    return [HIDDEN_DIM]