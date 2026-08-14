import torch
import torch.nn as nn

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self):
        super().__init__()
        self.scale = 0.125  # 1/sqrt(64) where 64 is the head dimension

    def forward(self, query, key, value):
        # Compute attention scores: scale the query key dot product
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, value)
        
        return output

# Module-level constants for shapes
BATCH_SIZE = 8
SEQ_LEN = 128
NUM_HEADS = 8
HEAD_DIM = 64

def get_inputs():
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    return []