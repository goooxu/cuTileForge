import torch
import torch.nn as nn

# Module-level constants for shape configuration
BATCH_SIZE = 2
SEQ_LEN = 4096
NUM_HEADS = 16
HEAD_DIM = 64

class Model(nn.Module):
    """SomeName (tier 5, matmul)"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value):
        # Scale the queries by the square root of head dimension
        scale = torch.sqrt(torch.tensor(HEAD_DIM, dtype=torch.float32))
        scaled_query = query / scale
        
        # Compute attention scores: (B, H, Q, D) @ (B, H, D, K) -> (B, H, Q, K)
        scores = torch.matmul(scaled_query, key.transpose(-2, -1))
        
        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values: (B, H, Q, K) @ (B, H, K, D) -> (B, H, Q, D)
        output = torch.matmul(attn_weights, value)
        
        return output

def get_inputs():
    """Generate input tensors for the model."""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    """Return empty list since __init__ takes no arguments."""
    return []