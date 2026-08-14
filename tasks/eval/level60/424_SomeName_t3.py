import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, elementwise)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, x):
        # Chain of elementwise operations
        x = torch.relu(x)
        x = torch.sigmoid(x)
        x = torch.tanh(x)
        x = torch.sqrt(torch.abs(x) + 1e-6)
        return x

INPUT_SIZE = [1, 2, 3, 4]
HIDDEN_SIZE = [5, 6, 7, 8]

def get_inputs():
    return [torch.randn(INPUT_SIZE)]

def get_init_inputs():
    return []
_EVAL_MARK = 1
