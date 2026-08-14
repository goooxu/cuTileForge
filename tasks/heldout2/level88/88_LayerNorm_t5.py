import torch
import torch.nn as nn

"""LayerNorm (tier 5, norm)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 8
SEQ_LEN = 1024
HIDDEN_DIM = 768

class Model(nn.Module):
    """LayerNorm (tier 5, norm)"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.layernorm(x)

def get_inputs():
    # Create input tensor with shape (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)]

def get_init_inputs():
    # Return arguments for __init__: normalized_shape and eps
    return [HIDDEN_DIM, 1e-5]