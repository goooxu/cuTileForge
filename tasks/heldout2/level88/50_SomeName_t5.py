import torch
import torch.nn as nn

"""SomeName (tier 5, elementwise)"""


class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""

    def __init__(self, hidden_size, eps=1e-5):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        
        # LayerNorm is used for normalization
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        
        # Register a buffer for the residual weight to avoid randomness
        self.register_buffer('residual_weight', torch.ones(1))

    def forward(self, input_tensor, residual_tensor):
        # Normalize the input
        normalized = self.layer_norm(input_tensor)
        
        # Add residual connection
        added = normalized + residual_tensor
        
        # Apply activation (ReLU)
        output = torch.relu(added)
        
        return output


# Module-level constants for shapes
INPUT_BATCH_SIZE = 4
INPUT_SEQ_LENGTH = 8
INPUT_HIDDEN_SIZE = 16

RESIDUAL_BATCH_SIZE = 4
RESIDUAL_SEQ_LENGTH = 8
RESIDUAL_HIDDEN_SIZE = 16


def get_inputs():
    """Return input tensors for the forward pass"""
    input_tensor = torch.randn(INPUT_BATCH_SIZE, INPUT_SEQ_LENGTH, INPUT_HIDDEN_SIZE)
    residual_tensor = torch.randn(RESIDUAL_BATCH_SIZE, RESIDUAL_SEQ_LENGTH, RESIDUAL_HIDDEN_SIZE)
    return [input_tensor, residual_tensor]


def get_init_inputs():
    """Return arguments for __init__"""
    return [INPUT_HIDDEN_SIZE]