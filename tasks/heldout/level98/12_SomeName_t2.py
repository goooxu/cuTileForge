import torch
import torch.nn as nn

"""
SomeName (tier 2, matmul)
"""

INPUT_SIZE = 4
HIDDEN_SIZE = 8
OUTPUT_SIZE = 3
BATCH_SIZE = 5

class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        # Define the layer - note that despite the docstring saying "conv",
        # this is a matmul-based implementation following the specification
        self.linear = nn.Linear(input_size, output_size)
        # Add bias by setting bias=True in Linear layer
        # Activation: ReLU
        self.activation = nn.ReLU()

    def forward(self, x):
        # Matrix multiplication: x @ W^T + b (done by nn.Linear)
        # followed by activation
        x = self.linear(x)
        x = self.activation(x)
        return x


def get_inputs():
    """Return a list of tensors to pass to forward."""
    return [torch.randn(BATCH_SIZE, INPUT_SIZE)]


def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]