import torch
import torch.nn as nn

# Module-level constants for large tensor shapes
INPUT_CHANNELS = 1024
BATCH_SIZE = 64
HEIGHT = 64
WIDTH = 64

class Model(nn.Module):
    """RMSNormLayer (tier 5, norm)"""

    def __init__(self, num_features, eps=1e-6):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.eval()  # Set to eval mode for deterministic behavior

    def forward(self, x):
        # RMSNorm computation: x / sqrt(mean(x^2) + eps) * weight
        # Reshape for proper broadcasting
        original_shape = x.shape
        x_reshaped = x.view(original_shape[0], -1, self.num_features)
        
        # Compute mean of squared values along feature dimension
        rms = torch.sqrt(torch.mean(x_reshaped ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        normalized = x_reshaped / rms * self.weight
        
        # Reshape back to original shape
        output = normalized.view(original_shape)
        return output

def get_inputs():
    # Create large tensor for throughput measurement
    return [torch.randn(BATCH_SIZE, INPUT_CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # Return configuration for module initialization
    return [INPUT_CHANNELS]