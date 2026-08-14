import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormExample (tier 3, norm)"""
    
    def __init__(self, num_features, eps=1e-5, elementwise_affine=True):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.layer_norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        return self.layer_norm(x)

# Module-level constants for shape configuration
BATCH_SIZE = 4
SEQUENCE_LENGTH = 1024
NUM_FEATURES = 512

def get_inputs():
    """Create input tensor for LayerNorm"""
    return [torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, NUM_FEATURES)]

def get_init_inputs():
    """Create initialization arguments for LayerNorm"""
    return [NUM_FEATURES, 1e-5, True]