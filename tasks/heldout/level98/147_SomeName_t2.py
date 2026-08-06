import torch
import torch.nn as nn

"""SomeName (tier 2, matmul)"""


class Model(nn.Module):
    """SomeName (tier 2, conv)"""

    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Linear layer for matrix multiplication
        self.linear = nn.Linear(input_size, output_size, bias=True)
        
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        return x


# Module-level constants for shapes
INPUT_SIZE = 256
HIDDEN_SIZE = 128
OUTPUT_SIZE = 64
BATCH_SIZE = 32

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_SIZE)]

def get_init_inputs():
    return [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]