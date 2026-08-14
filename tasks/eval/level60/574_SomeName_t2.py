import torch
import torch.nn as nn

"""SomeName (tier 2, matmul)"""

# Module-level constants for shapes
INPUT_DIM = 8
HIDDEN_DIM = 12
OUTPUT_DIM = 10

class Model(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Matrix multiplication weight
        self.weight = nn.Parameter(torch.randn(hidden_dim, input_dim))
        
        # Bias for the linear transformation
        self.bias = nn.Parameter(torch.randn(hidden_dim))
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Matrix multiply: (batch, input_dim) @ (input_dim, hidden_dim) -> (batch, hidden_dim)
        # Then add bias and apply activation
        result = torch.matmul(x, self.weight.t()) + self.bias
        result = self.activation(result)
        return result

def get_inputs():
    # Create input tensor with batch size 2
    return [torch.randn(2, INPUT_DIM)]

def get_init_inputs():
    # Return the configuration parameters for __init__
    return [INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM]
_EVAL_MARK = 1
