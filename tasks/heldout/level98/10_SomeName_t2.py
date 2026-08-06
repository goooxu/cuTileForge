import torch
import torch.nn as nn

"""SomeName (tier 2, elementwise)"""

# Module-level constants for shapes
BATCH_SIZE = 4
SEQ_LEN = 4096
HIDDEN_DIM = 2048
EPSILON = 1e-5
IN_FEATURES = HIDDEN_DIM
OUT_FEATURES = HIDDEN_DIM

class Model(nn.Module):
    def __init__(self, in_features, out_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        
        # LayerNorm for normalization
        self.layer_norm = nn.LayerNorm(in_features, eps=eps)
        
        # Linear layer for projection (could be seen as a residual connection operation)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # ReLU activation
        self.activation = nn.ReLU()
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to ensure deterministic behavior."""
        nn.init.kaiming_uniform_(self.linear.weight, a=5**0.5)

    def forward(self, x):
        # Normalize input
        normalized = self.layer_norm(x)
        
        # Apply linear transformation
        projected = self.linear(normalized)
        
        # Apply activation
        activated = self.activation(projected)
        
        return activated

def get_inputs():
    """Return input tensor."""
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)]

def get_init_inputs():
    """Return arguments for __init__."""
    return [IN_FEATURES, OUT_FEATURES, EPSILON]