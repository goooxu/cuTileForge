import torch
import torch.nn as nn

# Shape constants
BATCH_SIZE = 8
SEQ_LEN = 512
NUM_HEADS = 16
HEAD_DIM = 64

class Model(nn.Module):
    """ScaledDotProductAttention (tier 3, matmul)"""

    def __init__(self, num_heads=16, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

    def forward(self, query, key, value):
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attention_weights, value)
        
        return output

def get_inputs():
    """Generate input tensors for the attention computation"""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float32)
    return [query, key, value]

def get_init_inputs():
    """Return initialization parameters for the model"""
    return [NUM_HEADS, HEAD_DIM]