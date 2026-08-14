import torch
import torch.nn as nn

class Model(nn.Module):
    """TransformerBlock (tier 2, elementwise)"""
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)
        
        # Multi-head attention (simplified as linear layers for this exercise)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Feed-forward network
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
        
        # For deterministic behavior, ensure no randomness
        self.eval()
    
    def forward(self, x):
        # Save input for residual connection
        residual = x
        
        # Apply layer normalization
        normalized = self.norm(x)
        
        # Multi-head attention (simplified)
        q = self.q_proj(normalized)
        k = self.k_proj(normalized)
        v = self.v_proj(normalized)
        
        # Scaled dot-product attention (simplified)
        attn_output = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attn_output = torch.softmax(attn_output, dim=-1)
        attn_output = torch.matmul(attn_output, v)
        attn_output = self.out_proj(attn_output)
        
        # Add residual
        x = attn_output + residual
        
        # Save for second residual connection
        residual2 = x
        
        # Apply layer normalization again
        normalized2 = self.norm(x)
        
        # Feed-forward network
        out = self.fc1(normalized2)
        out = self.activation(out)
        out = self.fc2(out)
        
        # Add residual
        x = out + residual2
        
        return x

# Module-level constants for shapes
HIDDEN_SIZE = 512
NUM_HEADS = 8
INTERMEDIATE_SIZE = 2048
BATCH_SIZE = 4
SEQ_LENGTH = 64

def get_inputs():
    """Returns list of tensors to pass to forward"""
    x = torch.randn(BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE)
    return [x]

def get_init_inputs():
    """Returns list of arguments to pass to __init__"""
    return [HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE]