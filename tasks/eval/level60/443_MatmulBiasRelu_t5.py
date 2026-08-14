import torch
import torch.nn as nn

class Model(nn.Module):
    """MatmulBiasRelu (tier 5, matmul)"""

    def __init__(self, in_features, hidden_features, out_features):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Module-level constants for tensor shapes
IN_FEATURES = 128
HIDDEN_FEATURES = 256
OUT_FEATURES = 64
BATCH_SIZE = 48
def get_inputs():
    """Returns a list of tensors to pass to forward."""
    return [torch.randn(BATCH_SIZE, IN_FEATURES)]

def get_init_inputs():
    """Returns a list of arguments to pass to __init__."""
    return [IN_FEATURES, HIDDEN_FEATURES, OUT_FEATURES]