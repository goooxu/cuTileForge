import torch
import torch.nn as nn

class Model(nn.Module):
    """LayerNormExample (tier 5, norm)"""

    def __init__(self, normalized_shape=(128, 64), eps=1e-5):
        super(Model, self).__init__()
        self.norm_layer = nn.LayerNorm(normalized_shape, eps=eps)
        self.norm_layer.eval()  # Ensure deterministic behavior

    def forward(self, x):
        return self.norm_layer(x)

# Module-level constants for shapes
INPUT_HEIGHT = 64
INPUT_WIDTH = 128
BATCH_SIZE = 32
CHANNELS = 1

def get_inputs():
    return [torch.randn(BATCH_SIZE, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32)]

def get_init_inputs():
    return [(INPUT_HEIGHT, INPUT_WIDTH), 1e-5]