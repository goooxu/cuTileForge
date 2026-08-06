import torch
import torch.nn as nn

"""SomeName (tier 3, matmul)"""


class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    def __init__(self, input_features, output_features, bias=True):
        super(Model, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.has_bias = bias
        
        # Create linear layer (matrix multiply + bias)
        self.linear = nn.Linear(input_features, output_features, bias=bias)
        
        # Add batch norm and set to eval mode for deterministic behavior
        self.bn = nn.BatchNorm1d(output_features)
        self.bn.eval()
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Matrix multiply (via linear layer) + bias
        out = self.linear(x)
        
        # Batch norm
        out = self.bn(out)
        
        # Activation
        out = self.relu(out)
        
        return out


# Module-level constants for shapes
INPUT_FEATURES = 256
OUTPUT_FEATURES = 512
BATCH_SIZE = 32


def get_inputs():
    """Return list of tensors to pass to forward method"""
    return [torch.randn(BATCH_SIZE, INPUT_FEATURES)]


def get_init_inputs():
    """Return list of arguments to pass to __init__"""
    return [INPUT_FEATURES, OUTPUT_FEATURES]