import torch
import torch.nn as nn

"""SomeName (tier 5, conv)"""

M = 256
K = 512
N = 128


class Model(nn.Module):
    """SomeName (tier 5, conv)"""
    
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.relu = nn.ReLU()
        
        # Initialize with deterministic values
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
            if self.linear.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.linear.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(self.linear.bias, -bound, bound)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x


def get_inputs():
    return [torch.randn(M, K)]


def get_init_inputs():
    return [K, N, True]


import math