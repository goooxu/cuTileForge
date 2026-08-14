import torch
import torch.nn as nn

class Model(nn.Module):
    """AttentionScoreComputation (tier 3, matmul)"""

    def __init__(self, seq_len, num_heads, head_dim):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
    def forward(self, query, key, value):
        # Compute attention scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Compute output: (batch, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, value)
        
        return output

# Module-level constants for shapes
BATCH_SIZE = 6
SEQ_LEN = 1024
NUM_HEADS = 16
HEAD_DIM = 64

def get_inputs():
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    return [SEQ_LEN, NUM_HEADS, HEAD_DIM]