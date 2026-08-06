import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self, in_channels, hidden_dim, kernel_size=3, stride=1, padding=1):
        super(Model, self).__init__()
        
        # Store dimensions
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # Main convolution path
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.norm.eval()  # Make deterministic
        
        # Residual path for matching dimensions
        self.residual = nn.Conv2d(in_channels, hidden_dim, 1)
        
        # Activation
        self.act = nn.ReLU(inplace=False)  # Non-inplace to avoid modifying inputs

    def forward(self, x):
        # Main path
        main = self.conv1(x)
        main = self.norm(main)
        
        # Residual path
        residual = self.residual(x)
        
        # Add residual and activation
        out = main + residual
        out = self.act(out)
        
        return out


# Module-level constants for tensor sizes
N = 4
C_IN = 16
H = 64
W = 64
C_HIDDEN = 32

def get_inputs():
    return [
        torch.randn(N, C_IN, H, W),
    ]

def get_init_inputs():
    return [
        C_IN,  # in_channels
        C_HIDDEN,  # hidden_dim
    ]