import torch
import torch.nn as nn

"""SomeName (norm, elementwise)"""

NORM_FEATURES = 4
RES_FEATURES = 4

class Model(nn.Module):
    """SomeName (tier 2, conv)"""
    def __init__(self, norm_features=NORM_FEATURES, res_features=RES_FEATURES):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_features, affine=False)
        self.res_weight = nn.Parameter(torch.randn(res_features, 1, 1) * 0.1)
        self.activation = nn.ReLU()
        self.norm.eval()
        
    def forward(self, x, residual):
        x = self.norm(x)
        x = x + self.res_weight * residual
        x = self.activation(x)
        return x

def get_inputs():
    x = torch.randn(1, NORM_FEATURES, 3, 3)
    residual = torch.randn(1, RES_FEATURES, 3, 3)
    return [x, residual]

def get_init_inputs():
    return [NORM_FEATURES, RES_FEATURES]