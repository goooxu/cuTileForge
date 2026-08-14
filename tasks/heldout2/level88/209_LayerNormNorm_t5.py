import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormNorm (tier 5, norm)"""
    
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        return self.layer_norm(x)

# Module-level constants for shape configuration
NORMALIZED_SHAPE = (4, 4)
EPS = 1e-5
ELEMENTWISE_AFFINE = True

def get_inputs():
    """Returns input tensors for the model"""
    return [torch.randn(2, 4, 4)]

def get_init_inputs():
    """Returns initialization arguments for the model"""
    return [NORMALIZED_SHAPE, EPS, ELEMENTWISE_AFFINE]