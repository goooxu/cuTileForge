import torch
import torch.nn as nn

class Model(nn.Module):
    """Softplus (tier 3, norm)"""

    def __init__(self, normalized_shape, eps=1e-4, elementwise_affine=True):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.layer_norm.eval()

    def forward(self, x):
        return torch.nn.functional.softplus(self.layer_norm(x))

# Module-level constants for shape configuration
NORMALIZED_SHAPE = (64, 128, 256)
EPS = 1e-5
ELEMENTWISE_AFFINE = True

def get_inputs():
    # Create input tensor with shape matching normalized_shape
    # Adding batch dimension for medium-sized tensor
    batch_size = 8
    input_shape = (batch_size, 64, 128, 256)
    return [torch.randn(input_shape)]

def get_init_inputs():
    return [NORMALIZED_SHAPE, EPS, ELEMENTWISE_AFFINE]