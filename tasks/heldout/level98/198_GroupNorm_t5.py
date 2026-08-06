import torch
import torch.nn as nn

# Module-level constants for shape definitions
N = 2  # batch size
C = 8  # number of channels
H = 4  # height
W = 4  # width

class Model(nn.Module):
    """GroupNorm (tier 5, norm)"""
    
    def __init__(self, num_groups=2, num_channels=8):
        super(Model, self).__init__()
        # Use GroupNorm with deterministic behavior
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.norm.eval()  # Ensure deterministic behavior for evaluation
    
    def forward(self, x):
        return self.norm(x)

def get_inputs():
    # Create input tensor of shape (N, C, H, W) = (2, 8, 4, 4)
    return [torch.randn(N, C, H, W)]

def get_init_inputs():
    # Return initialization arguments matching the expected parameters
    return [2, 8]