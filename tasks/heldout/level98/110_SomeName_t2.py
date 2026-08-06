import torch
import torch.nn as nn

"""SomeName (tier 2, matmul)"""

INPUT_FEATURES = 4
OUTPUT_FEATURES = 4
BATCH_SIZE = 1
DROPOUT_RATE = 0.1

class Model(nn.Module):
    """SomeName (tier 2, matmul)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(INPUT_FEATURES, OUTPUT_FEATURES, bias=True)
        self.activation = nn.ReLU()
    
    def forward(self, input_tensor):
        output = self.linear(input_tensor)
        output = self.activation(output)
        return output


def get_inputs():
    return [
        torch.randn(BATCH_SIZE, INPUT_FEATURES),
    ]


def get_init_inputs():
    return []