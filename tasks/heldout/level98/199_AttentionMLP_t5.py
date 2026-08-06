import torch
import torch.nn as nn

"""AttentionMLP (tier 5, elementwise)"""

class Model(nn.Module):
    """AttentionMLP (tier 5, conv)"""
    
    def __init__(self, embed_size=128, seq_len=32, num_heads=4, head_dim=32):
        super(Model, self).__init__()
        self.embed_size = embed_size
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Affine transforms for query, key, value
        self.W_q = nn.Linear(embed_size, head_dim * num_heads, bias=False)
        self.W_k = nn.Linear(embed_size, head_dim * num_heads, bias=False)
        self.W_v = nn.Linear(embed_size, head_dim * num_heads, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(head_dim * num_heads, embed_size, bias=False)
        
        # Set to eval mode for deterministic behavior
        self.eval()
    
    def forward(self, inputs):
        """
        Compute attention scores, apply softmax, and perform weighted sum.
        This is an attention-style score computation with scale, softmax, and matmul.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_size)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, embed_size)
        """
        batch_size, seq_len, embed_size = inputs.shape
        
        # Linear projections
        query = self.W_q(inputs)  # (batch, seq_len, head_dim * num_heads)
        key = self.W_k(inputs)    # (batch, seq_len, head_dim * num_heads)
        value = self.W_v(inputs)  # (batch, seq_len, head_dim * num_heads)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Permute to (batch, num_heads, seq_len, head_dim) for attention computation
        query = query.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        key = key.permute(0, 2, 1, 3)      # (batch, num_heads, seq_len, head_dim)
        value = value.permute(0, 2, 1, 3)  # (batch, num_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        # 1. Scale: divide by sqrt(head_dim) for stability
        scale = 1.0 / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        
        # 2. Compute attention scores: (batch, num_heads, seq_len, seq_len)
        #    matmul(query, key^T) = matmul(query, key.transpose(-2, -1))
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # 3. Apply softmax along the last dimension (sequence dimension)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 4. Apply attention weights to values: matmul(attn_weights, value)
        #    result shape: (batch, num_heads, seq_len, head_dim)
        context = torch.matmul(attn_weights, value)
        
        # Permute back to (batch, seq_len, num_heads, head_dim)
        context = context.permute(0, 2, 1, 3).contiguous()
        
        # Reshape to (batch, seq_len, num_heads * head_dim)
        context = context.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # Final linear projection
        output = self.W_o(context)
        
        return output


# Module-level constants for shapes
BATCH_SIZE = 8
SEQ_LEN = 32
EMBED_SIZE = 128
NUM_HEADS = 4
HEAD_DIM = 32

def get_inputs():
    """
    Return input tensors for forward pass.
    
    Returns:
        list: List containing one tensor of shape (BATCH_SIZE, SEQ_LEN, EMBED_SIZE)
    """
    return [torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_SIZE)]

def get_init_inputs():
    """
    Return arguments to pass to __init__.
    
    Returns:
        list: List of arguments for model initialization
    """
    return [EMBED_SIZE, SEQ_LEN, NUM_HEADS, HEAD_DIM]