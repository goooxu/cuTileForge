import torch
import torch.nn as nn

class Model(nn.Module):
    """TransformerBlock (tier 5, conv)"""

    def __init__(self, hidden_size=4096, num_heads=16, intermediate_size=16384):
        super(Model, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        
        # Layer norms
        self.input_norm = nn.LayerNorm(hidden_size)
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
        # Convolutional projection for attention (simulating transformer block)
        self.query_conv = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_heads)
        self.key_conv = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_heads)
        self.value_conv = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_heads)
        self.output_conv = nn.Conv1d(hidden_size, hidden_size, 1, groups=num_heads)
        
        # Feed-forward network with convolution
        self.ffn_conv1 = nn.Conv1d(hidden_size, intermediate_size, 1)
        self.ffn_conv2 = nn.Conv1d(intermediate_size, hidden_size, 1)
        
        # Evaluation mode for deterministic behavior
        self.input_norm.eval()
        self.attention_norm.eval()
        self.output_norm.eval()

    def forward(self, x):
        # x shape: (batch, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        
        # Residual connection for input
        residual = x
        
        # Layer normalization
        x = self.input_norm(x)
        
        # Reshape for 1D convolution (batch, hidden, seq_len)
        x = x.permute(0, 2, 1).contiguous()
        
        # Compute query, key, value projections
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        # Reshape for attention computation
        query = query.view(batch_size, self.num_heads, -1, seq_len)
        key = key.view(batch_size, self.num_heads, -1, seq_len)
        value = value.view(batch_size, self.num_heads, -1, seq_len)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attention_output = torch.matmul(attention_weights, value)
        attention_output = attention_output.view(batch_size, self.hidden_size, seq_len)
        
        # Project attention output
        attention_output = self.output_conv(attention_output)
        
        # Reshape back to (batch, seq_len, hidden)
        attention_output = attention_output.permute(0, 2, 1).contiguous()
        
        # First residual connection and normalization
        x = attention_output + residual
        x = self.attention_norm(x)
        
        # Second residual connection for FFN
        residual = x
        
        # FFN with convolution
        x = x.permute(0, 2, 1).contiguous()
        x = self.ffn_conv1(x)
        x = torch.relu(x)
        x = self.ffn_conv2(x)
        x = x.permute(0, 2, 1).contiguous()
        
        # Final residual connection and normalization
        x = x + residual
        x = self.output_norm(x)
        
        return x

# Module-level constants for shapes
BATCH_SIZE = 2
SEQ_LEN = 1024
HIDDEN_SIZE = 4096
NUM_HEADS = 16
INTERMEDIATE_SIZE = 16384

def get_inputs():
    # Return a list with one tensor matching the expected input shape
    return [torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)]

def get_init_inputs():
    # Return arguments for __init__ matching the configuration
    return [HIDDEN_SIZE, NUM_HEADS, INTERMEDIATE_SIZE]