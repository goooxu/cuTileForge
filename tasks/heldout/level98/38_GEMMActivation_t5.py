import torch
import torch.nn as nn

class Model(nn.Module):
    """GEMMActivation (tier 5, matmul)"""

    def __init__(self, in_features, hidden_features, out_features, bias=True):
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features, bias=bias)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

IN_FEATURES = 256
HIDDEN_FEATURES = 512
OUT_FEATURES = 128
BATCH_SIZE = 64

def get_inputs():
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    return [x]

def get_init_inputs():
    return [IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES, True]