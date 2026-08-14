import torch
import torch.nn as nn

# Module-level constants for shapes
BATCH_SIZE = 4
SEQ_LEN = 32
NUM_HEADS = 8
HEAD_DIM = 16
HIDDEN_DIM = NUM_HEADS * HEAD_DIM

class Model(nn.Module):
    """AttentionScoreComputation (tier 3, matmul)"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, query, key, value):
        """
        Compute attention scores: scale(query @ key^T), softmax, then matmul with value.
        
        Args:
            query: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
            key: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
            value: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
            
        Returns:
            output: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        """
        # Scaled dot-product attention
        # 1. Compute attention scores: query @ key^T
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 2. Scale the scores
        scale_factor = 1.0 / torch.sqrt(torch.tensor(HEAD_DIM, dtype=torch.float32))
        scaled_scores = scores * scale_factor
        
        # 3. Apply softmax along the last dimension (key sequence dimension)
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # 4. Compute final output: attention_weights @ value
        output = torch.matmul(attention_weights, value)
        
        return output

def get_inputs():
    """Generate input tensors for the attention computation."""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    return [query, key, value]

def get_init_inputs():
    """Return arguments for __init__ (empty list for this model)."""
    return []