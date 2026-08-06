import torch
import torch.nn as nn

class Model(nn.Module):
    """NormActRes (tier 2, norm)"""

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # Layer normalization for pre-norm transformer block
        self.norm = nn.LayerNorm(hidden_size, eps=eps)
        
        # Initialize as eval to be deterministic
        self.norm.eval()

    def forward(self, x):
        # Pre-norm transformer tail: normalise, residual, activation
        # x: [batch_size, seq_len, hidden_size]
        # Compute layernorm (deterministic)
        x = self.norm(x)
        return x


# Module-level constants for shapes
BATCH_SIZE = 4
SEQ_LEN = 32
HIDDEN_SIZE = 64


def get_inputs():
    """Return list of input tensors for forward pass."""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
    return [x]


def get_init_inputs():
    """Return list of arguments for __init__."""
    return [HIDDEN_SIZE, 1e-5]