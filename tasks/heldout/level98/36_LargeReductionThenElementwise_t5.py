import torch
import torch.nn as nn

class Model(nn.Module):
    """LargeReductionThenElementwise (tier 5, reduction)"""

    def __init__(self, input_shape):
        super().__init__()
        # Mark module as BatchNorm-aware, but we only use Conv for actual computation
        # Ensure no randomness in forward pass
        self.input_shape = input_shape
        # Conv1d will be used to create deterministic initial state
        self.initial_conv = nn.Conv1d(input_shape[1], 1, 1, bias=False)
        # Mark BatchNorm modules as always in eval mode
        self.bn1 = nn.BatchNorm1d(input_shape[1])
        self.bn1.eval()

    def forward(self, x):
        # First do batchnorm (eval mode already)
        x = self.bn1(x)
        # Reduce along last axis: torch.sum(x, dim=-1)
        reduced = torch.sum(x, dim=-1)
        # Then apply convolution to reduced result
        # Expand dim to make it (B, C, 1) for 1D conv
        reduced_expanded = reduced.unsqueeze(-1)
        conv_out = self.initial_conv(reduced_expanded)
        # Remove extra dim to return (B, C_out)
        return conv_out.squeeze(-1)


INPUT_BATCH = 64
INPUT_FEATURES = 512
INPUT_LENGTH = 4096
INIT_INPUT_SHAPE = [INPUT_BATCH, INPUT_FEATURES, INPUT_LENGTH]

def get_inputs():
    # Create deterministic inputs
    torch.manual_seed(42)
    x = torch.randn(INPUT_BATCH, INPUT_FEATURES, INPUT_LENGTH)
    return [x]

def get_init_inputs():
    return [INIT_INPUT_SHAPE]