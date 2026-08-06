import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 2, elementwise)"""

    def __init__(self):
        super(Model, self).__init__()
        # No learnable parameters needed for elementwise operations
        pass
    
    def forward(self, input1, input2, input3, input4):
        # Chain of 5 elementwise operations:
        # 1. Add: input1 + input2
        # 2. Subtract: (input1 + input2) - input3
        # 3. Multiply: ((input1 + input2) - input3) * input4
        # 4. Divide: (((input1 + input2) - input3) * input4) / 2.0
        # 5. Add final: (((input1 + input2) - input3) * input4) / 2.0 + 1.0
        result = (input1 + input2) - input3
        result = result * input4
        result = result / 2.0
        result = result + 1.0
        return result

# Module-level constants for shapes
INPUT1_SHAPE = (10000, 1000)
INPUT2_SHAPE = (10000, 1000)
INPUT3_SHAPE = (10000, 1000)
INPUT4_SHAPE = (10000, 1000)

def get_inputs():
    """Returns list of tensors for forward pass"""
    input1 = torch.ones(INPUT1_SHAPE, dtype=torch.float32)
    input2 = torch.ones(INPUT2_SHAPE, dtype=torch.float32)
    input3 = torch.ones(INPUT3_SHAPE, dtype=torch.float32)
    input4 = torch.ones(INPUT4_SHAPE, dtype=torch.float32) * 0.5
    return [input1, input2, input3, input4]

def get_init_inputs():
    """Returns arguments for __init__"""
    return []