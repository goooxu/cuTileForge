import torch
import torch.nn as nn

"""AdaptiveAvgPool1D (tier 3, pool)"""

class Model(nn.Module):
    """AdaptiveAvgPool1D (tier 3, pool)"""

    def __init__(self, output_size):
        super(Model, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return torch.nn.functional.adaptive_avg_pool1d(x, self.output_size)

INPUT_SEQ_LEN = 1024 * 256
BATCH_SIZE = 1
INPUT_CHANNELS = 256
OUTPUT_SIZE = 1

def get_inputs():
    input_tensor = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_SEQ_LEN, dtype=torch.float32)
    return [input_tensor]

def get_init_inputs():
    return [OUTPUT_SIZE]