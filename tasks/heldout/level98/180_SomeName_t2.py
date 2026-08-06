import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, reduction)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
        # BatchNorm2d is evaluated at runtime, so mark as eval for deterministic behavior
        self.norm = nn.BatchNorm2d(4)
        self.norm.eval()
        
        # Pre-compute indices for deterministic gather operation
        # Using fixed indices for the reduction
        self.register_buffer('_gather_indices', torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long))
    
    def forward(self, x):
        # First dimension: batch size, second: channels, third and fourth: spatial dimensions
        # Normalize the input first
        x = self.norm(x)
        
        # Reduce along the channel dimension (dim=1) by summing
        # Then expand to original shape for elementwise operation
        reduced = x.sum(dim=1, keepdim=True)  # shape: (batch_size, 1, H, W)
        
        # Expand the reduced tensor to match original shape for elementwise multiplication
        expanded = reduced.expand(-1, x.size(1), -1, -1)
        
        # Elementwise multiplication with original input
        result = x * expanded
        
        return result

# Shape constants
BATCH_SIZE = 2
CHANNELS = 4
HEIGHT = 64
WIDTH = 64

def get_inputs():
    # Return a single tensor input to forward
    return [torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)]

def get_init_inputs():
    # No inputs needed for __init__ as it uses default parameters
    return []