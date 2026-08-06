import torch
import torch.nn as nn

"""
ScaleDotAttention (tier 3, matmul)
"""

# Module-level constants for tensor shapes
BATCH_SIZE = 2
SEQ_LEN = 4
HEAD_DIM = 8
NUM_HEADS = 1

class Model(nn.Module):
    """SomeName (tier 3, conv)"""
    
    def __init__(self, dropout_p=0.0):
        super().__init__()
        self.scale = (HEAD_DIM * NUM_HEADS) ** -0.5
        
    def forward(self, query, key, value):
        # query, key: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        # value: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        
        # Scaled dot-product attention score computation
        # 1. Scaled matmul of query and key
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # 2. Softmax over the last dimension (key sequence dimension)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 3. Weighted sum of values
        output = torch.matmul(attn_weights, value)
        
        return output

def get_inputs():
    """Generate deterministic input tensors"""
    query = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.ones(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    """Generate initialization arguments"""
    return [0.0]  # dropout_p parameter