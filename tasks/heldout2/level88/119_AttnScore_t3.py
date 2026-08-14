import torch
import torch.nn as nn

# Shape constants
BATCH_SIZE = 4
SEQ_LEN = 2048
NUM_HEADS = 16
HEAD_DIM = 64
D_MODEL = NUM_HEADS * HEAD_DIM

class Model(nn.Module):
    """AttnScore (tier 3, matmul)"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value):
        # Scale: Q @ K^T / sqrt(HEAD_DIM)
        scale = HEAD_DIM ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Softmax along the last dimension (key positions)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Final matmul: attn_weights @ V
        output = torch.matmul(attn_weights, value)
        
        return output

def get_inputs():
    # Generate large tensors for attention computation
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    return []