import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""

    def __init__(self, input_size=256):
        super(Model, self).__init__()
        self.input_size = input_size

    def forward(self, x):
        # Chain of 4+ elementwise operations
        x = x * 2.0
        x = x + 1.0
        x = x ** 2
        x = torch.sqrt(x)
        return x


INPUT_SIZE = 256

def get_inputs():
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    return [INPUT_SIZE]
_EVAL_MARK = 1
