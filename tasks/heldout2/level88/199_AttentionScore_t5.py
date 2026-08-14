import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 8
SEQ_LEN = 128
NUM_HEADS = 16
HEAD_DIM = 64
EMBED_DIM = NUM_HEADS * HEAD_DIM

class Model(nn.Module):
    """AttentionScore (tier 5, matmul)"""
    
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)
        
    def forward(self, query, key, value):
        # Compute attention scores
        # query: (B, H, S, D), key: (B, H, D, S), value: (B, H, S, D)
        # Result: (B, H, S, S)
        
        # Scaled dot-product: (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply softmax along the last dimension
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values: (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        output = torch.matmul(attn_weights, value)
        
        return output

def get_inputs():
    # Create input tensors with shapes suitable for attention computation
    # Query, Key, Value: (batch_size, num_heads, seq_len, head_dim)
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    # Return arguments for __init__
    return [NUM_HEADS, HEAD_DIM]