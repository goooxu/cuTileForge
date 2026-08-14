import torch
import torch.nn as nn

# Shape constants for attention computation
BATCH_SIZE = 3
SEQ_LEN = 8
HEAD_DIM = 16
NUM_HEADS = 4

class Model(nn.Module):
    """SomeName (tier 2, matmul)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, query, key, value):
        # Compute attention scores with scaling
        scale = 1.0 / torch.sqrt(torch.tensor(HEAD_DIM, dtype=torch.float32))
        
        # Batch matrix multiplication: (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the scores
        scores = scores * scale
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output: (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        output = torch.matmul(attention_weights, value)
        
        return output

def get_inputs():
    """Create input tensors for the attention computation"""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    """Return empty list since __init__ doesn't need arguments"""
    return []