import torch
import torch.nn as nn

class Model(nn.Module):
    """RMSNorm (tier 2, norm)"""

    def __init__(self, num_features, eps=1e-5):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        # Create learnable scale parameter
        self.weight = nn.Parameter(torch.ones(num_features))
        
    def forward(self, x):
        # Compute RMS normalization
        # RMS = sqrt(mean(x^2))
        # normalized = x / RMS * weight
        original_shape = x.shape
        x = x.view(-1, self.num_features)
        
        # Compute mean of squares along feature dimension
        sq_mean = torch.mean(x * x, dim=1, keepdim=True)
        
        # Compute RMS (root mean square)
        rms = torch.sqrt(sq_mean + self.eps)
        
        # Normalize and scale
        normalized = x / rms * self.weight
        
        return normalized.view(original_shape)

# Module-level constants for shape configuration
BATCH_SIZE = 32
SEQ_LEN = 512
NUM_FEATURES = 1024

def get_inputs():
    # Return a list with a single tensor of appropriate size
    return [torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)]

def get_init_inputs():
    # Return arguments for __init__
    return [NUM_FEATURES]