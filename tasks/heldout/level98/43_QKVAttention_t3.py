import torch
import torch.nn as nn

"""QKVAttention (tier 3, matmul)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 2
SEQ_LEN = 4
NUM_HEADS = 2
HEAD_DIM = 8
EMBED_DIM = HEAD_DIM * NUM_HEADS

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # Pre-computed constant scaling factor: 1 / sqrt(HEAD_DIM)
        self.scale = 1.0 / (HEAD_DIM ** 0.5)
        
    def forward(self, query, key, value):
        # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the scores
        scaled_scores = scores * self.scale
        
        # Apply softmax along the last dimension (sequence dimension)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # Apply attention weights to values: attention_weights @ V
        output = torch.matmul(attention_weights, value)
        
        return output

def get_inputs():
    # Generate deterministic inputs for the attention computation
    # Shape: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    query = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    
    # Add small deterministic offset to make values non-zero and avoid numerical issues
    query = query + 0.01
    key = key + 0.02
    value = value + 0.03
    
    return [query, key, value]

def get_init_inputs():
    # No initialization arguments needed for this model
    return []