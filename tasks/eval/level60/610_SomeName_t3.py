import torch
import torch.nn as nn

# Module-level constants for shape configuration
BATCH_SIZE = 48
SEQ_LEN = 512
HEADS = 16
HEAD_DIM = 64

class Model(nn.Module):
    """SomeName (tier 3, matmul)"""

    def __init__(self):
        super().__init__()
        self.scale = 1.0 / (HEAD_DIM ** 0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # Compute attention scores: Q @ K^T
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Scale the scores
        scaled_scores = scores * self.scale
        
        # Apply softmax
        attention_weights = self.softmax(scaled_scores)
        
        # Compute output: attention_weights @ V
        output = torch.matmul(attention_weights, value)
        
        return output


def get_inputs():
    """Generate input tensors for the attention computation"""
    query = torch.randn(BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]


def get_init_inputs():
    """Return initialization arguments for the model"""
    return []