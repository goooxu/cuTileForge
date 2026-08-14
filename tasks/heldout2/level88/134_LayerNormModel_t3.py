import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormModel (tier 3, norm)"""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.layer_norm.eval()  # Ensure deterministic behavior

    def forward(self, x):
        return self.layer_norm(x)

# Shape constants
BATCH_SIZE = 32
SEQ_LEN = 128
HIDDEN_DIM = 768

def get_inputs():
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)]

def get_init_inputs():
    return [HIDDEN_DIM]