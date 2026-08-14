import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormNorm (tier 2, norm)"""

    def __init__(self, normalized_shape, eps=1e-5):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x):
        return self.layer_norm(x)

# Module-level constants for shape configuration
BATCH_SIZE = 32
SEQ_LEN = 512
HIDDEN_DIM = 1024

def get_inputs():
    """Return input tensor for LayerNorm"""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)]

def get_init_inputs():
    """Return initialization parameters for LayerNorm"""
    return [HIDDEN_DIM, 1e-5]