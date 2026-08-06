import torch
import torch.nn as nn

"""SomeName (tier 3, matmul)"""

# Module-level constants for tensor shapes
BATCH_SIZE = 32
SEQ_LEN = 512
NUM_HEADS = 32
HEAD_DIM = 64

class Model(nn.Module):
    """SomeName (tier 3, matmul)"""
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, query, key, value):
        # Compute attention scores: scale, softmax, then matmul
        # query: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        # key:   (BATCH_SIZE, NUM_HEADS, HEAD_DIM, SEQ_LEN)
        # value: (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
        
        # Scale the query by sqrt(HEAD_DIM)
        scale = torch.sqrt(torch.tensor(HEAD_DIM, dtype=torch.float32))
        scaled_query = query / scale
        
        # Compute attention scores: matmul of scaled_query and key
        scores = torch.matmul(scaled_query, key)
        
        # Apply softmax along the last dimension (SEQ_LEN)
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attention_weights, value)
        
        return output

def get_inputs():
    """Return list of tensors for forward pass"""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, HEAD_DIM, SEQ_LEN)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    """Return list of arguments for __init__"""
    return []