import torch
import torch.nn as nn

"""SomeName (tier 3, matmul)"""

# Module-level constants for shape configuration
BATCH_SIZE = 16
M = 64
K = 128
N = 64

class Model(nn.Module):
    def __init__(self, input_dim=None, bias=True):
        super(Model, self).__init__()
        self.input_dim = input_dim or K
        self.output_dim = N
        
        # Linear layer performs matrix multiply + bias
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=bias)
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Set to eval mode for deterministic behavior
        self.linear.eval()
        self.activation.eval()
    
    def forward(self, x):
        # Matrix multiply: x @ W^T + b, then activation
        out = self.linear(x)
        out = self.activation(out)
        return out

def get_inputs():
    # Create input tensor of shape (batch_size, K)
    x = torch.randn(BATCH_SIZE, K)
    return [x]

def get_init_inputs():
    # Return configuration parameters for Model initialization
    return [K]