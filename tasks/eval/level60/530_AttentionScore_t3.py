import torch
import torch.nn as nn

class Model(nn.Module):
    """AttentionScore (tier 3, conv)"""

    def __init__(self, query_size, key_size, value_size):
        super(Model, self).__init__()
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        
    def forward(self, query, key, value):
        # Compute attention scores: scale, softmax, then matmul
        # Scale by sqrt(key_size)
        scale_factor = 1.0 / torch.sqrt(torch.tensor(self.key_size, dtype=torch.float32))
        scaled_query = query * scale_factor
        
        # Compute attention scores (batch_size, seq_len, key_size) @ (batch_size, key_size, seq_len)
        # Result: (batch_size, seq_len, seq_len)
        attention_scores = torch.matmul(scaled_query, key.transpose(-2, -1))
        
        # Apply softmax along the last dimension
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values: (batch_size, seq_len, seq_len) @ (batch_size, seq_len, value_size)
        # Result: (batch_size, seq_len, value_size)
        output = torch.matmul(attention_weights, value)
        
        return output


# Module-level constants for shapes
BATCH_SIZE = 3
SEQ_LEN = 4
QUERY_SIZE = 8
KEY_SIZE = 8
VALUE_SIZE = 8

def get_inputs():
    """Generate input tensors for the attention model"""
    query = torch.randn(BATCH_SIZE, SEQ_LEN, QUERY_SIZE)
    key = torch.randn(BATCH_SIZE, SEQ_LEN, KEY_SIZE)
    value = torch.randn(BATCH_SIZE, SEQ_LEN, VALUE_SIZE)
    return [query, key, value]

def get_init_inputs():
    """Return initialization parameters for the model"""
    return [QUERY_SIZE, KEY_SIZE, VALUE_SIZE]