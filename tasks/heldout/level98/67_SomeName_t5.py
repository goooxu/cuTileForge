import torch
import torch.nn as nn

"""SomeName (tier 5, matmul)"""

# Module-level constants for tensor shapes
INPUT_DIM = 8192
HIDDEN_DIM = 8192
OUTPUT_DIM = 8192
BATCH_SIZE = 1

class Model(nn.Module):
    """SomeName (tier 5, matmul)"""
    
    def __init__(self):
        super().__init__()
        # Create a linear layer (matrix multiply + bias)
        self.linear = nn.Linear(INPUT_DIM, OUTPUT_DIM, bias=True)
        # Use ReLU as the activation
        self.activation = nn.ReLU()
        # Ensure deterministic behavior for inference
        self.linear.eval()
    
    def forward(self, input_tensor):
        # Matrix multiply + bias from linear layer
        x = self.linear(input_tensor)
        # Apply activation function
        x = self.activation(x)
        return x

def get_inputs():
    """Generate input tensor for the model."""
    return [torch.randn(BATCH_SIZE, INPUT_DIM)]

def get_init_inputs():
    """Return arguments for model initialization."""
    return []