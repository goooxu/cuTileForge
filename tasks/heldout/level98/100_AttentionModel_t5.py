import torch
import torch.nn as nn

# Shape constants for medium tensors
BATCH_SIZE = 4
SEQ_LENGTH = 64
NUM_HEADS = 8
HEAD_DIM = 16

class Model(nn.Module):
    """AttentionModel (tier 5, conv)"""

    def __init__(self, num_heads=NUM_HEADS, head_dim=HEAD_DIM):
        super(Model, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.sqrt_head_dim = head_dim ** 0.5
        
        # Linear layers for Q, K, V projections (simulating attention)
        self.q_proj = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.k_proj = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        self.v_proj = nn.Linear(head_dim * num_heads, head_dim * num_heads)
        
        # Initialize weights with constant values for deterministic behavior
        nn.init.constant_(self.q_proj.weight, 0.1)
        nn.init.constant_(self.k_proj.weight, 0.1)
        nn.init.constant_(self.v_proj.weight, 0.1)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)

    def forward(self, x):
        """
        Compute attention scores: scale, softmax, then matmul.
        Args:
            x: Input tensor of shape (batch_size, seq_length, num_heads * head_dim)
        Returns:
            Tensor of shape (batch_size, seq_length, num_heads * head_dim)
        """
        batch_size, seq_length, embed_dim = x.shape
        
        # Project to Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_length, embed_dim)
        K = self.k_proj(x)  # (batch_size, seq_length, embed_dim)
        V = self.v_proj(x)  # (batch_size, seq_length, embed_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, H, D)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, H, D)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim)  # (B, S, H, D)
        
        # Permute to (batch_size, num_heads, seq_length, head_dim)
        Q = Q.permute(0, 2, 1, 3)  # (B, H, S, D)
        K = K.permute(0, 2, 1, 3)  # (B, H, S, D)
        V = V.permute(0, 2, 1, 3)  # (B, H, S, D)
        
        # Compute attention scores
        # (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_head_dim
        
        # Apply softmax (only along the last dimension)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights to values
        # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        context = torch.matmul(attn_weights, V)
        
        # Reshape back to original dimensions
        context = context.permute(0, 2, 1, 3).contiguous()  # (B, S, H, D)
        context = context.view(batch_size, seq_length, -1)  # (B, S, H*D)
        
        return context

def get_inputs():
    """Return a list of tensors to pass to the forward method."""
    # Create a deterministic input tensor
    x = torch.ones(BATCH_SIZE, SEQ_LENGTH, NUM_HEADS * HEAD_DIM)
    return [x]

def get_init_inputs():
    """Return a list of arguments to pass to __init__."""
    return [NUM_HEADS, HEAD_DIM]