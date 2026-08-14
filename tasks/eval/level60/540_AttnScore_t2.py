import torch
import torch.nn as nn

# Shape constants
BATCH_SIZE = 24
SEQ_LENGTH = 1024
NUM_HEADS = 16
HEAD_DIM = 64

class Model(nn.Module):
    """AttnScore (tier 2, matmul)"""
    
    def __init__(self):
        super().__init__()
        self.scale = 1.0 / (HEAD_DIM ** 0.5)
    
    def forward(self, query, key):
        # Compute attention scores: scale the query, then matmul with key
        scaled_query = query * self.scale
        scores = torch.matmul(scaled_query, key.transpose(-2, -1))
        return scores

def get_inputs():
    # Generate input tensors for attention computation
    # Query shape: (BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM)
    # Key shape: (BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM)
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, HEAD_DIM)
    return [query, key]

def get_init_inputs():
    # No initialization arguments needed
    return []