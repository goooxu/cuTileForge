import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x):
        # Chain of elementwise operations on tensor x
        y1 = x * 2.0
        y2 = y1 + 1.0
        y3 = torch.relu(y2)
        y4 = y3 / (torch.abs(y3) + 1.0)
        return y4

INPUT_SIZE = [64, 64, 64]

def get_inputs():
    x = torch.randn(*INPUT_SIZE)
    return [x]

def get_init_inputs():
    return []
_EVAL_MARK = 1
