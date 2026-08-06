import torch
import torch.nn as nn

"""SomeName (tier 2, norm)"""


class Model(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # BatchNorm2d module
        self.norm = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)
        
        # Call eval() to make forward deterministic
        self.norm.eval()
        
        # Residual connection is direct identity mapping (no learnable projection)
    
    def forward(self, x):
        # Apply normalization
        normalized = self.norm(x)
        
        # Residual add (x + normalized)
        residual = x + normalized
        
        # Activation (ReLU)
        output = torch.relu(residual)
        
        return output


# Module-level constants for shape configuration
BATCH_SIZE = 32
NUM_CHANNELS = 256
HEIGHT = 112
WIDTH = 112

# Use large but reasonable tensor size for throughput measurement
# This will be ~32 * 256 * 112 * 112 * 4 bytes ≈ 41 MB per tensor
INPUT_SHAPE = (BATCH_SIZE, NUM_CHANNELS, HEIGHT, WIDTH)


def get_inputs():
    """Return list of tensors to pass to forward"""
    return [torch.randn(INPUT_SHAPE, dtype=torch.float32)]


def get_init_inputs():
    """Return list of arguments to pass to __init__"""
    return [NUM_CHANNELS, 1e-5, 0.1]