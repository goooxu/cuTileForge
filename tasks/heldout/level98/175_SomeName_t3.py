import torch
import torch.nn as nn

"""
TransformerBlock (tier 3, elementwise)
"""


class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self, in_channels=1024, hidden_channels=2048):
        super(Model, self).__init__()
        # Conv1d layer with kernel size 1 for efficient computation
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.ReLU()
        
        # Set to eval mode for deterministic behavior
        self.norm.eval()
        
        # Initialize weights for consistency
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        residual = x
        out = self.conv1(x)
        out = self.act(out)
        out = self.conv2(out)
        out = out + residual
        out = out.permute(0, 2, 1)  # (batch, seq_len, channels)
        out = self.norm(out)
        out = out.permute(0, 2, 1)  # (batch, channels, seq_len)
        return out


# Shape constants
BATCH_SIZE = 4
IN_CHANNELS = 1024
SEQ_LEN = 2048

def get_inputs():
    x = torch.randn(BATCH_SIZE, IN_CHANNELS, SEQ_LEN)
    return [x]

def get_init_inputs():
    return []