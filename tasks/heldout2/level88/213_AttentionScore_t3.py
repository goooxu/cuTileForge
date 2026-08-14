import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 2
SEQ_LEN = 4
NUM_HEADS = 2
HEAD_DIM = 8

class Model(nn.Module):
    """AttentionScore (tier 3, matmul)"""
    
    def __init__(self, scale_factor=1.0):
        super(Model, self).__init__()
        self.scale_factor = scale_factor
    
    def forward(self, query, key, value):
        # Compute attention scores: scale, softmax, then matmul
        # query: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        # key: (BATCH_SIZE, NUM_HEADS, HEAD_DIM, SEQ_LEN)
        # value: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        
        # Step 1: Compute raw attention scores (matmul of query and key)
        attn_scores = torch.matmul(query, key)  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, SEQ_LEN)
        
        # Step 2: Scale the scores
        scaled_scores = attn_scores * self.scale_factor
        
        # Step 3: Apply softmax to get attention weights
        attn_weights = torch.softmax(scaled_scores, dim=-1)
        
        # Step 4: Compute final output by matmul with values
        output = torch.matmul(attn_weights, value)  # (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        
        return output

def get_inputs():
    # Create input tensors with deterministic values
    query = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.ones(BATCH_SIZE, NUM_HEADS, HEAD_DIM, SEQ_LEN)
    value = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    # Return scale factor as initialization parameter
    return [1.0 / (HEAD_DIM ** 0.5)]