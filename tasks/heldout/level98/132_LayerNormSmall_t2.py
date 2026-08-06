import torch
import torch.nn as nn

"""LayerNormSmall (tier 2, norm)"""

INPUT_CHANNEL = 4
INPUT_HEIGHT = 3
INPUT_WIDTH = 3

class Model(nn.Module):
    """LayerNormSmall (tier 2, norm)"""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(Model, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.layernorm.eval()  # Ensure deterministic behavior

    def forward(self, x):
        return self.layernorm(x)

INPUT_CHANNEL = 4
INPUT_HEIGHT = 3
INPUT_WIDTH = 3

def get_inputs():
    # Create tensor with shape (batch_size=2, channels=4, height=3, width=3)
    x = torch.randn(2, INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH)
    return [x]

def get_init_inputs():
    # normalized_shape for LayerNorm expects the last N dimensions to normalize
    # For input (2, 4, 3, 3), we normalize over channels, height, width: normalized_shape=(4, 3, 3)
    return [(INPUT_CHANNEL, INPUT_HEIGHT, INPUT_WIDTH)]