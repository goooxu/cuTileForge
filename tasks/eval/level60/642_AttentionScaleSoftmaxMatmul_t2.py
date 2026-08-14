import torch
import torch.nn as nn

# Module-level constants for shape configuration
BATCH_SIZE = 6
SEQ_LEN = 8
NUM_HEADS = 2
HEAD_DIM = 16

class Model(nn.Module):
    """AttentionScaleSoftmaxMatmul (tier 2, matmul)"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, query, key, value):
        # Compute attention scores: (batch, heads, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale by square root of head dimension
        scaled_scores = scores / torch.sqrt(torch.tensor(HEAD_DIM, dtype=torch.float32))
        
        # Apply softmax along the last dimension
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # Compute final output: (batch, heads, seq_len, head_dim)
        output = torch.matmul(attention_weights, value)
        
        return output

def get_inputs():
    """Generate input tensors for the attention model."""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    """Return empty list since __init__ takes no arguments."""
    return []