import torch
import torch.nn as nn

# Module-level constants for shapes
BATCH_SIZE = 3
SEQ_LEN = 4
HEAD_DIM = 8
NUM_HEADS = 1

class Model(nn.Module):
    """AttentionScore (tier 5, matmul)"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key):
        # Compute attention scores: scale, softmax, then matmul
        # query: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        # key: (BATCH_SIZE, NUM_HEADS, HEAD_DIM, SEQ_LEN)
        
        # Scale the query-key product
        scale = 1.0 / (HEAD_DIM ** 0.5)
        scaled_scores = torch.matmul(query, key) * scale
        
        # Apply softmax along the last dimension (sequence positions)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # Return the attention weights as the final output
        return attention_weights

def get_inputs():
    # Create deterministic inputs for the model
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_DIM, SEQ_LEN)
    return [query, key]

def get_init_inputs():
    # No initialization inputs needed
    return []