import torch
import torch.nn as nn

# Module-level constants for shapes
BATCH_SIZE = 12
SEQ_LEN = 1024
NUM_HEADS = 16
HEAD_DIM = 64

class Model(nn.Module):
    """AttentionScoreComputation (tier 2, matmul)"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value, scale):
        """
        Compute attention scores: scale -> softmax -> matmul
        
        Args:
            query: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
            key: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
            value: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
            scale: scalar for scaling the attention scores
            
        Returns:
            output: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        """
        # Compute attention scores: Q @ K^T
        # query: [B, H, L, D], key: [B, H, L, D]
        # key_t: [B, H, D, L]
        key_t = key.transpose(-1, -2)  # [B, H, D, L]
        scores = torch.matmul(query, key_t)  # [B, H, L, L]
        
        # Scale the scores
        scores = scores * scale
        
        # Apply softmax along the last dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute output: attention_weights @ V
        output = torch.matmul(attention_weights, value)  # [B, H, L, D]
        
        return output

def get_inputs():
    """Return input tensors for the model"""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    scale = torch.tensor(1.0 / (HEAD_DIM ** 0.5))  # Standard attention scaling
    return [query, key, value, scale]

def get_init_inputs():
    """Return arguments for model initialization"""
    return []