import torch
import torch.nn as nn

"""SomeName (tier 5, matmul)"""

# Module-level constants for shape dimensions
INPUT_DIM = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 64

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Matrix multiplication weights
        self.weight = nn.Parameter(torch.randn(INPUT_DIM, HIDDEN_DIM))
        # Bias for the linear transformation
        self.bias = nn.Parameter(torch.randn(HIDDEN_DIM))
        # ReLU activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x @ weight adds bias and then applies ReLU
        # Matrix multiply: (batch, INPUT_DIM) @ (INPUT_DIM, HIDDEN_DIM) -> (batch, HIDDEN_DIM)
        # Add bias element-wise: (batch, HIDDEN_DIM) + (HIDDEN_DIM) -> (batch, HIDDEN_DIM)
        # Apply activation function: (batch, HIDDEN_DIM)
        output = torch.mm(x, self.weight) + self.bias
        output = self.relu(output)
        return output

def get_inputs():
    # Create input tensor of shape (batch_size, INPUT_DIM)
    # Using a fixed batch size of 16 for medium tensor operation
    batch_size = 16
    x = torch.randn(batch_size, INPUT_DIM)
    return [x]

def get_init_inputs():
    # No initialization parameters needed since we're using default construction
    return []