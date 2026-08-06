import torch
import torch.nn as nn


class Model(nn.Module):
    """GroupNorm (tier 3, norm)"""

    def __init__(self, num_features: int, num_groups: int):
        super(Model, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)

    def forward(self, x: torch.Tensor):
        return self.gn(x)


batch_size = 16
features = 16
dim1 = 32
dim2 = 128
num_groups = 4

def get_inputs():
    return [torch.rand(batch_size, features, dim1, dim2)]


def get_init_inputs():
    return [features, num_groups]
