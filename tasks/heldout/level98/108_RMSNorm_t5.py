import torch
import torch.nn as nn

"""RMSNorm (tier 5, norm)"""

# Shape constants
BATCH_SIZE = 64
SEQ_LEN = 1024
HIDDEN_DIM = 4096

class Model(nn.Module):
    """RMSNorm (tier 5, norm)"""

    def __init__(self, hidden_dim, eps=1e-6):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        # RMSNorm: root mean square normalization
        # x: (batch_size, seq_len, hidden_dim)
        # Compute RMS: sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
        x_norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normalized = x * x_norm
        # Apply learnable weight
        output = x_normalized * self.weight
        return output


def get_inputs():
    """Return input tensor for forward pass"""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM, dtype=torch.float32)]


def get_init_inputs():
    """Return arguments for model initialization"""
    return [HIDDEN_DIM]