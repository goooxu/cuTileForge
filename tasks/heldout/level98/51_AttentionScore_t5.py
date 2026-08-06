import torch
import torch.nn as nn

class Model(nn.Module):
    """AttentionScore (tier 5, conv)"""

    def __init__(self, batch_size, seq_len, d_model, num_heads):
        super().__init__()
        # Precompute attention scale
        self.scale = 1.0 / torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        # Initialize a dummy BatchNorm for eval() call
        self.bn = nn.BatchNorm1d(d_model)
        self.bn.eval()

    def forward(self, query, key, value):
        # Compute attention scores: (batch_size, num_heads, seq_len, d_model)
        # Scale the query keys
        scaled_query = query * self.scale
        
        # Compute score matrix: (batch_size, num_heads, seq_len, seq_len)
        scores = torch.bmm(
            scaled_query.transpose(1, 2).reshape(self.batch_size * self.num_heads, self.seq_len, self.d_model),
            key.transpose(1, 2).reshape(self.batch_size * self.num_heads, self.d_model, self.seq_len)
        )
        scores = scores.reshape(self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        
        # Apply softmax along the sequence length dimension
        attention_weights = torch.softmax(scores, dim=-1)
        
        # Compute weighted sum of values
        output = torch.bmm(
            attention_weights.reshape(self.batch_size * self.num_heads, self.seq_len, self.seq_len),
            value.transpose(1, 2).reshape(self.batch_size * self.num_heads, self.seq_len, self.d_model)
        )
        output = output.reshape(self.batch_size, self.num_heads, self.seq_len, self.d_model)
        
        return output


# Module-level constants for shape configuration
BATCH_SIZE = 4
SEQ_LEN = 512
D_MODEL = 768
NUM_HEADS = 12


def get_inputs():
    """Generate input tensors for the attention computation"""
    query = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_MODEL, dtype=torch.float32)
    key = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_MODEL, dtype=torch.float32)
    value = torch.randn(BATCH_SIZE, NUM_HEADS, SEQ_LEN, D_MODEL, dtype=torch.float32)
    return [query, key, value]


def get_init_inputs():
    """Return arguments for __init__"""
    return [BATCH_SIZE, SEQ_LEN, D_MODEL, NUM_HEADS]