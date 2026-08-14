import torch
import torch.nn as nn

INPUT_SIZE = 32
HIDDEN_SIZE = 64
OUTPUT_SIZE = 128


class Model(nn.Module):
    """SomeName (tier 5, elementwise)"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Chain of 5 elementwise operations
        # 1. Square
        x = x * x
        # 2. Multiply by 2
        x = x * 2.0
        # 3. Add bias
        x = x + 1.0
        # 4. Take absolute value
        x = torch.abs(x)
        # 5. Sigmoid-like operation: 1 / (1 + exp(-x))
        x = 1.0 / (1.0 + torch.exp(-x))
        return x


def get_inputs():
    return [torch.randn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)]


def get_init_inputs():
    return []
_EVAL_MARK = 1
