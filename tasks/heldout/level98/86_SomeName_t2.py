import torch
import torch.nn as nn

"""SomeName (tier 2, elementwise)"""

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""

    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_features)
        self.norm.eval()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = x * 2.0
        x = torch.exp(x)
        x = torch.log(x + 1.0)
        x = torch.sqrt(x + 1.0)
        x = x - 0.5
        return x


# Module-level constants for shapes
IN_FEATURES = 256
OUT_FEATURES = 256
BATCH_SIZE = 128
HEIGHT = 56
WIDTH = 56

def get_inputs():
    return [
        torch.empty(BATCH_SIZE, IN_FEATURES, HEIGHT, WIDTH, dtype=torch.float32, requires_grad=False),
    ]

def get_init_inputs():
    return [
        IN_FEATURES,
        OUT_FEATURES,
    ]