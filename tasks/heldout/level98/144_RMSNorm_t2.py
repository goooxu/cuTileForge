import torch
import torch.nn as nn

"""RMSNorm (tier 2, norm)"""


class Model(nn.Module):
    def __init__(self, feature_dim=1024, eps=1e-5):
        super(Model, self).__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        # Create learned scale parameter
        self.scale = nn.Parameter(torch.ones(feature_dim))

    def forward(self, x):
        # Compute RMS normalization
        # x: [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
        # We apply normalization along the last dimension
        if x.dim() == 2:
            # [batch_size, feature_dim]
            x_norm = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        elif x.dim() == 3:
            # [batch_size, seq_len, feature_dim]
            x_norm = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        else:
            raise ValueError(f"Unexpected input dimensions: {x.dim()}")
        
        # Apply learned scale
        x_out = x_norm * self.scale
        
        return x_out


# Module-level constants for shape configuration
BATCH_SIZE = 64
SEQ_LEN = 512
FEATURE_DIM = 1024


def get_inputs():
    """Return list of tensors for forward pass."""
    # Create input tensor with shape [batch_size, seq_len, feature_dim]
    # Using larger values to ensure good throughput measurement
    x = torch.randn(BATCH_SIZE, SEQ_LEN, FEATURE_DIM, requires_grad=True)
    return [x]


def get_init_inputs():
    """Return list of arguments for __init__."""
    return [FEATURE_DIM]