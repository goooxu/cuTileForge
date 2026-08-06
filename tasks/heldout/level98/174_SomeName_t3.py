import torch
import torch.nn as nn

"""SomeName (tier 3, norm)"""
class Model(nn.Module):
    """SomeName (tier 3, norm)"""
    def __init__(self, hidden_size=128):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        
        # Initialize with specific weights for determinism
        self.norm.weight.data.fill_(1.0)
        self.norm.bias.data.zero_()
        
        self.eval()

    def forward(self, x):
        return self.norm(x)


INPUT_BATCH = 8
INPUT_SEQ_LEN = 64
INPUT_HIDDEN = 128

def get_inputs():
    x = torch.ones(INPUT_BATCH, INPUT_SEQ_LEN, INPUT_HIDDEN)
    return [x]

def get_init_inputs():
    return [INPUT_HIDDEN]