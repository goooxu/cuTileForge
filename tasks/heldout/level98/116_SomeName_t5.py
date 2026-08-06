import torch
import torch.nn as nn

"""SomeName (tier 5, norm)"""

# Module-level constants
BATCH_SIZE = 8
SEQ_LEN = 128
HIDDEN_DIM = 768

class Model(nn.Module):
    def __init__(self, hidden_dim, eps=1e-5):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.layer_norm.eval()  # Make deterministic by disabling training mode
    
    def forward(self, x):
        x_norm = self.layer_norm(x)
        x = x + x_norm  # Residual connection
        x = torch.relu(x)  # Activation
        return x

def get_inputs():
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)]

def get_init_inputs():
    return [HIDDEN_DIM]