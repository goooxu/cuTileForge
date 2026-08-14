import torch
import torch.nn as nn

class Model(nn.Module):
    """ReductionSum (tier 2, reduction)"""
    
    def __init__(self, batch_size, seq_len, hidden_dim):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        # Reduction along the sequence axis (dim=1)
        reduced = x.sum(dim=1)
        
        # Elementwise operations: scale and add bias
        result = reduced * 0.5 + 1.0
        
        return result

# Module-level constants for tensor shapes
BATCH_SIZE = 192
SEQ_LEN = 1024
HIDDEN_DIM = 2048

def get_inputs():
    """Return list of input tensors"""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_DIM)
    return [x]

def get_init_inputs():
    """Return list of arguments to pass to __init__"""
    return [BATCH_SIZE, SEQ_LEN, HIDDEN_DIM]