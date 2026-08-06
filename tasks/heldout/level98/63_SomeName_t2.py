import torch
import torch.nn as nn

# Module-level constants for shape configuration
BATCH_SIZE = 128
IN_CHANNELS = 64
OUT_CHANNELS = 128
HEIGHT = 32
WIDTH = 32

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        self.register_buffer('alpha', torch.tensor(0.5))
        self.register_buffer('beta', torch.tensor(1.2))
        self.register_buffer('gamma', torch.tensor(-0.3))
        self.register_buffer('delta', torch.tensor(2.0))
        self.register_buffer('epsilon', torch.tensor(1.0))

    def forward(self, x):
        # Chain of 5 elementwise operations
        out = x * self.alpha
        out = torch.abs(out)
        out = out + self.beta
        out = torch.tanh(out)
        out = out * self.gamma
        out = torch.sigmoid(out)
        out = out * self.delta + self.epsilon
        return out

def get_inputs():
    return [
        torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)
    ]

def get_init_inputs():
    return []