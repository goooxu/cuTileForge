import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, matmul)"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc = nn.Linear(input_dim, hidden_dim, bias=True)
        self.activation = nn.ReLU()
        
        # Initialize weights deterministically
        nn.init.constant_(self.fc.weight, 0.5)
        nn.init.constant_(self.fc.bias, 0.1)
    
    def forward(self, x):
        out = self.fc(x)
        out = self.activation(out)
        return out

# Module-level constants for shape configuration
INPUT_DIM = 8
HIDDEN_DIM = 12
OUTPUT_DIM = 12

def get_inputs():
    """Return input tensors for the model."""
    return [torch.randn(4, INPUT_DIM)]

def get_init_inputs():
    """Return initialization arguments for the model."""
    return [INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM]
_EVAL_MARK = 1
