import torch
import torch.nn as nn

"""GlobalAveragePool3D (tier 2, pool)"""

NAMESPACES = {
    "batch_size": 2,
    "channels": 3,
    "depth": 4,
    "height": 5,
    "width": 6
}

class Model(nn.Module):
    """GlobalAveragePool3D (tier 2, pool)"""
    
    def __init__(self):
        super(Model, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    
    def forward(self, x):
        return self.pool(x)

def get_inputs():
    return [
        torch.randn(
            NAMESPACES["batch_size"],
            NAMESPACES["channels"],
            NAMESPACES["depth"],
            NAMESPACES["height"],
            NAMESPACES["width"]
        )
    ]

def get_init_inputs():
    return []