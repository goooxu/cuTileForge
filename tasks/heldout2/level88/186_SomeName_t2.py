import torch
import torch.nn as nn

"""SomeName (tier 2, matmul)"""

# Module-level constants for shapes
INPUT_FEATURES = 8
OUTPUT_FEATURES = 8
BATCH_SIZE = 4

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Define the matrix multiplication and bias addition
        self.linear = nn.Linear(INPUT_FEATURES, OUTPUT_FEATURES, bias=True)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Matrix multiply followed by bias and activation
        x = self.linear(x)
        x = self.activation(x)
        return x

def get_inputs():
    # Return input tensor with shape (BATCH_SIZE, INPUT_FEATURES)
    return [torch.randn(BATCH_SIZE, INPUT_FEATURES)]

def get_init_inputs():
    # No additional initialization inputs needed
    return []