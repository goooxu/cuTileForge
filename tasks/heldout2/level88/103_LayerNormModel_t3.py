import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormModel (tier 3, norm)"""

    def __init__(self, normalized_shape, eps=1e-5):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        # Ensure deterministic behavior
        self.layer_norm.eval()

    def forward(self, x):
        return self.layer_norm(x)

# Module-level constants for shape configuration
BATCH_SIZE = 4
SEQ_LEN = 1024
HIDDEN_DIM = 4096

def get_inputs():
    """Returns a list containing a single input tensor."""
    # Create a tensor with shape (BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [x]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    # Return the normalized_shape as a tuple
    return [(HIDDEN_DIM,)]