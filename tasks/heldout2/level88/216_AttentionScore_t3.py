import torch
import torch.nn as nn

class Model(nn.Module):
    """AttentionScore (tier 3, matmul)"""
    
    def __init__(self, batch_size, seq_len, num_heads, head_dim):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
    def forward(self, query, key, value):
        # Compute attention scores
        # Scale
        scale = self.head_dim ** -0.5
        query = query * scale
        
        # Compute attention weights (softmax)
        # Transpose key for matrix multiplication
        key_t = key.transpose(-1, -2)
        attn_weights = torch.matmul(query, key_t)
        
        # Apply softmax along the last dimension
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, value)
        
        return output

# Module-level constants for shapes
BATCH_SIZE = 2
SEQ_LEN = 4
NUM_HEADS = 3
HEAD_DIM = 8

def get_inputs():
    """Return list of input tensors for forward pass"""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return [BATCH_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]