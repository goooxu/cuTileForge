import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormNorm (tier 3, norm)"""

    def __init__(self, normalized_shape, eps=1e-4, elementwise_affine=True):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        return self.layer_norm(x)

# Module-level constants for shapes
NORMALIZED_SHAPE = (4, 5)
EPS = 1e-5
ELEMENTWISE_AFFINE = True

def get_inputs():
    # Create a tensor with shape compatible with LayerNorm
    # For LayerNorm, the last N dimensions (where N = len(normalized_shape)) should match normalized_shape
    # Here we create a tensor with shape (2, 3, 4, 5) to match normalized_shape (4, 5)
    x = torch.randn(2, 3, 4, 5)
    return [x]

def get_init_inputs():
    return [NORMALIZED_SHAPE, EPS, ELEMENTWISE_AFFINE]