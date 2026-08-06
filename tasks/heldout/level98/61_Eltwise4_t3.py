import torch
import torch.nn as nn

"""Eltwise4 (tier 3, elementwise)"""

# Module-level constants for shape specification
INPUT_SIZE = (64, 128, 32, 32)
SCALE_VAL = 0.5
ADD_VAL = 1.0
MULT_VAL = 2.0
LOG_VAL = 1.0

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self):
        super(Model, self).__init__()
        self.scale = nn.Parameter(torch.tensor(SCALE_VAL))
        self.add_val = ADD_VAL
        self.mult_val = MULT_VAL
        self.log_val = LOG_VAL
        
    def forward(self, x):
        # Chain of elementwise operations
        y1 = x * self.scale
        y2 = y1 + self.add_val
        y3 = torch.abs(y2)  # Ensures input to log is positive
        y4 = torch.mul(y3, self.mult_val)
        y5 = torch.log(y4 + self.log_val)
        y6 = torch.clamp(y5, min=-1.0, max=1.0)
        y7 = torch.exp(y6)
        y8 = y7 * y7
        return y8

def get_inputs():
    # Create deterministic input
    x = torch.ones(INPUT_SIZE, dtype=torch.float32)
    return [x]

def get_init_inputs():
    return []