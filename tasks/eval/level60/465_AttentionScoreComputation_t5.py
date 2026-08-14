import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 6
SEQ_LENGTH = 1024
HEADS = 16
HEAD_DIM = 64

class Model(nn.Module):
    """AttentionScoreComputation (tier 5, matmul)"""

    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value, scale_factor):
        # Compute attention scores: scale, softmax, then matmul
        # query, key, value have shape [BATCH_SIZE, HEADS, SEQ_LENGTH, HEAD_DIM]
        
        # Transpose key for matrix multiplication
        key_t = key.transpose(-1, -2)  # [BATCH_SIZE, HEADS, HEAD_DIM, SEQ_LENGTH]
        
        # Compute attention scores with scaling
        attn_scores = torch.matmul(query, key_t) * scale_factor
        
        # Apply softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Compute output via matmul with value
        output = torch.matmul(attn_weights, value)
        
        return output

def get_inputs():
    """Return input tensors for the attention computation"""
    query = torch.randn(BATCH_SIZE, HEADS, SEQ_LENGTH, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, HEADS, SEQ_LENGTH, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, HEADS, SEQ_LENGTH, HEAD_DIM)
    scale_factor = 1.0 / (HEAD_DIM ** 0.5)
    return [query, key, value, scale_factor]

def get_init_inputs():
    """Return initialization arguments"""
    return []