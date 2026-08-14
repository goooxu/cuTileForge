import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNorm (tier 5, norm)"""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        # Compute RMS normalization
        variance = x.pow(2).mean(-1, keepdim=True)
        normalized = x * torch.rsqrt(variance + self.eps)
        return normalized * self.weight


# Module-level constants for tensor shapes
INPUT_BATCH = 8
INPUT_SEQ_LEN = 128
HIDDEN_SIZE = 512

def get_inputs():
    return [
        torch.randn(INPUT_BATCH, INPUT_SEQ_LEN, HIDDEN_SIZE)
    ]

def get_init_inputs():
    return [HIDDEN_SIZE, 1e-6]
_EVAL_MARK = 1
