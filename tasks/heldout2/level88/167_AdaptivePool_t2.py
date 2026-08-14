import torch
import torch.nn as nn

class Model(nn.Module):
    """AdaptivePool (tier 2, pool)"""
    
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.output_size = output_size
        
    def forward(self, x):
        # Adaptive pooling to specified output size
        return nn.functional.adaptive_avg_pool1d(x, self.output_size)

# Module-level constants for shape configuration
INPUT_SIZE = 262144  # 2^18 elements
BATCH_SIZE = 1
OUTPUT_SIZE = 65536  # 2^16 elements for adaptive pooling

def get_inputs():
    """Generate input tensor for pooling operation"""
    x = torch.randn(BATCH_SIZE, INPUT_SIZE)
    return [x]

def get_init_inputs():
    """Return output size for adaptive pooling"""
    return [OUTPUT_SIZE]