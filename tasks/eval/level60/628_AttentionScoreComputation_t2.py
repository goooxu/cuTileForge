import torch
import torch.nn as nn

# Module-level constants for tensor shapes
BATCH_SIZE = 5
SEQ_LEN = 65
NUM_HEADS = 8
HEAD_DIM = 17
QUERY_DIM = NUM_HEADS * HEAD_DIM

class Model(nn.Module):
    """AttentionScoreComputation (tier 2, matmul)"""
    
    def __init__(self, query_dim, num_heads, head_dim, seq_len):
        super(Model, self).__init__()
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.seq_len = seq_len
        
        # Scale factor for attention (1/sqrt(head_dim))
        self.scale = 1.0 / (self.head_dim ** 0.5)
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj = nn.Linear(query_dim, query_dim, bias=False)
        self.v_proj = nn.Linear(query_dim, query_dim, bias=False)
    
    def forward(self, x):
        """
        Compute attention scores: scale, softmax, then matmul
        x: input tensor of shape (batch_size, seq_len, query_dim)
        """
        batch_size = x.shape[0]
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, query_dim)
        K = self.k_proj(x)  # (batch_size, seq_len, query_dim)
        V = self.v_proj(x)  # (batch_size, seq_len, query_dim)
        
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, self.seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, self.seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, self.seq_len, self.num_heads, self.head_dim)
        
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores: Q @ K^T
        K_T = K.transpose(-1, -2)  # (batch_size, num_heads, head_dim, seq_len)
        scores = torch.matmul(Q, K_T)  # (batch_size, num_heads, seq_len, seq_len)
        
        # Scale the scores
        scaled_scores = scores * self.scale
        
        # Apply softmax
        attention_weights = torch.softmax(scaled_scores, dim=-1)
        
        # Compute final output: attention_weights @ V
        output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Reshape back to (batch_size, seq_len, query_dim)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, self.seq_len, self.query_dim)
        
        return output

def get_inputs():
    """Generate input tensors for the model"""
    x = torch.randn(BATCH_SIZE, SEQ_LEN, QUERY_DIM)
    return [x]

def get_init_inputs():
    """Generate initialization arguments for the model"""
    return [QUERY_DIM, NUM_HEADS, HEAD_DIM, SEQ_LEN]