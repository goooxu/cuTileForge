import torch
import torch.nn as nn

# Module-level constants for shapes
BATCH_SIZE = 2
SEQ_LEN = 4
HEAD_DIM = 8

class Model(nn.Module):
    """SomeName (tier 3, conv)"""

    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value):
        # Compute attention scores: (B, S, H) @ (B, H, S) -> (B, S, S)
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale by sqrt(HEAD_DIM)
        scaled_scores = scores / torch.sqrt(torch.tensor(HEAD_DIM, dtype=torch.float32))
        
        # Apply softmax along the last dimension
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # Compute attention output: (B, S, S) @ (B, S, H) -> (B, S, H)
        output = torch.matmul(attention_weights, value)
        
        return output

def get_inputs():
    # Create input tensors with shapes (BATCH_SIZE, SEQ_LEN, HEAD_DIM)
    query = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    # No initialization arguments needed
    return []