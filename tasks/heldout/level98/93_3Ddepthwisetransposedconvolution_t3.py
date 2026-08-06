import torch
import torch.nn as nn

"""3D depthwise transposed convolution (tier 3, conv)"""

N, C, D, H, W = 1, 2, 4, 4, 4
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
dilation = 1

class Model(nn.Module):
    """3D depthwise transposed convolution (tier 3, conv)"""

    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels=C,
            out_channels=C,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=C,
            dilation=dilation,
            bias=False
        )
        self.conv.eval()

    def forward(self, x):
        return self.conv(x)

def get_inputs():
    x = torch.randn(N, C, D, H, W)
    return [x]

def get_init_inputs():
    return []