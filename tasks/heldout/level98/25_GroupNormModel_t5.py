import torch
import torch.nn as nn

class Model(nn.Module):
    """GroupNormModel (tier 5, norm)"""
    
    def __init__(self, num_groups=4, num_channels=128, eps=1e-5):
        super(Model, self).__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps)
        self.norm.eval()
    
    def forward(self, x):
        return self.norm(x)

# Module-level constants for shapes
BATCH_SIZE = 8
NUM_CHANNELS = 128
HEIGHT = 32
WIDTH = 32

def get_inputs():
    """Return input tensor for the model"""
    return [torch.randn(BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    """Return arguments for Model.__init__"""
    return [4, 128, 1e-5]