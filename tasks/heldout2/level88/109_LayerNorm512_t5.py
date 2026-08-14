import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNorm512 (tier 5, norm)"""

    def __init__(self, normalized_shape=(512,)):
        super(Model, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        return self.layer_norm(x)

# Module-level constants
INPUT_BATCH_SIZE = 16
INPUT_SEQ_LEN = 128
INPUT_FEATURES = 512

def get_inputs():
    return [torch.randn(INPUT_BATCH_SIZE, INPUT_SEQ_LEN, INPUT_FEATURES)]

def get_init_inputs():
    return [(INPUT_FEATURES,)]