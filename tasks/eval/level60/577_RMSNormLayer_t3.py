import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNormLayer (tier 3, norm)"""

    def __init__(self, num_features, eps=1e-6):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.eval()

    def forward(self, x):
        # RMS normalization: x / sqrt(mean(x^2) + eps) * weight
        x_sq = x * x
        mean_x_sq = x_sq.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_x_sq + self.eps)
        normalized = x / rms
        return normalized * self.weight


NUM_FEATURES = 4096
BATCH_SIZE = 48
SEQ_LENGTH = 1024

def get_inputs():
    return [torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES)]

def get_init_inputs():
    return [NUM_FEATURES]