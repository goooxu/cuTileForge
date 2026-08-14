import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 3
SEQ_LEN = 4
HEADS = 2
DIM_PER_HEAD = 3

class Model(nn.Module):
    """AttentionScore (tier 2, matmul)"""
    
    def __init__(self):
        super(Model, self).__init__()
        
    def forward(self, query, key, value):
        # Compute attention scores with scaling
        # query: (BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD)
        # key: (BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD)
        # value: (BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD)
        
        # Compute scaled attention scores: softmax(Q @ K^T / sqrt(d)) @ V
        scale_factor = DIM_PER_HEAD ** -0.5
        
        # Q @ K^T: (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
        attn_scores = torch.bmm(query, key.transpose(1, 2))
        
        # Scale: (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
        scaled_scores = attn_scores * scale_factor
        
        # Softmax: (BATCH_SIZE, SEQ_LEN, SEQ_LEN)
        attn_weights = torch.softmax(scaled_scores, dim=-1)
        
        # @ V: (BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD)
        output = torch.bmm(attn_weights, value)
        
        return output


def get_inputs():
    """Generate deterministic inputs for the attention computation."""
    query = torch.ones(BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD)
    key = torch.ones(BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD)
    value = torch.ones(BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD)
    
    # Add small deterministic perturbations to make the test meaningful
    query = query + torch.arange(BATCH_SIZE * SEQ_LEN * DIM_PER_HEAD, dtype=torch.float32).view(BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD) * 0.01
    key = key + torch.arange(BATCH_SIZE * SEQ_LEN * DIM_PER_HEAD, dtype=torch.float32).view(BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD) * 0.02
    value = value + torch.arange(BATCH_SIZE * SEQ_LEN * DIM_PER_HEAD, dtype=torch.float32).view(BATCH_SIZE, SEQ_LEN, DIM_PER_HEAD) * 0.03
    
    return [query, key, value]


def get_init_inputs():
    """Return empty list since __init__ takes no arguments."""
    return []