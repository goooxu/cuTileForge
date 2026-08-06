import torch
import torch.nn as nn

class Model(nn.Module):
    """Attention (tier 2, matmul)"""

    def __init__(self, query_len, key_len, hidden_dim, num_heads):
        super().__init__()
        self.query_len = query_len
        self.key_len = key_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = 1.0 / (self.hidden_dim // self.num_heads) ** 0.5
        
    def forward(self, query, key, value):
        # Compute attention scores: (batch, num_heads, query_len, key_len)
        # Using einsum for clarity: query (B, H, Q, D), key (B, H, D, K) -> scores (B, H, Q, K)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply softmax to attention scores
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values: (B, H, Q, K) * (B, H, K, D) -> (B, H, Q, D)
        output = torch.matmul(attn_weights, value)
        
        return output

# Module-level constants for shapes
BATCH_SIZE = 2
NUM_HEADS = 16
QUERY_LEN = 1024
KEY_LEN = 1024
HIDDEN_DIM = 1024

def get_inputs():
    """Generate input tensors for the attention computation."""
    # Query tensor: (batch, num_heads, query_len, hidden_dim_per_head)
    query = torch.randn(BATCH_SIZE, NUM_HEADS, QUERY_LEN, HIDDEN_DIM // NUM_HEADS)
    # Key tensor: (batch, num_heads, key_len, hidden_dim_per_head)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, KEY_LEN, HIDDEN_DIM // NUM_HEADS)
    # Value tensor: (batch, num_heads, key_len, hidden_dim_per_head)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, KEY_LEN, HIDDEN_DIM // NUM_HEADS)
    return [query, key, value]

def get_init_inputs():
    """Return arguments for Model initialization."""
    return [QUERY_LEN, KEY_LEN, HIDDEN_DIM, NUM_HEADS]