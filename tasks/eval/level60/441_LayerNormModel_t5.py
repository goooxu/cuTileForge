import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormModel (tier 5, norm)"""
    
    def __init__(self, normalized_shape, eps=1e-4, elementwise_affine=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        return self.layernorm(x)

# Module-level constants for shape configuration
BATCH_SIZE = 6
SEQ_LENGTH = 512
HIDDEN_DIM = 4096

def get_inputs():
    """Return a list of tensors to pass to forward."""
    return [torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [HIDDEN_DIM]