import torch
import torch.nn as nn

class Model(nn.Module):
    """MaxPool3D (tier 3, pool)"""

    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1):
        super(Model, self).__init__()
        self.pool = nn.MaxPool3d(kernel_size=kernel_size, 
                                 stride=stride, 
                                 padding=padding, 
                                 dilation=dilation)
        self.pool.eval()

    def forward(self, x):
        return self.pool(x)

# Shape constants
BATCH_SIZE = 1
IN_CHANNELS = 3
DEPTH = 128
HEIGHT = 128
WIDTH = 128

def get_inputs():
    return [torch.randn(BATCH_SIZE, IN_CHANNELS, DEPTH, HEIGHT, WIDTH)]

def get_init_inputs():
    return [2, 2, 0, 1]