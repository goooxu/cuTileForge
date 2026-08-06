import torch
import torch.nn as nn

class Model(nn.Module):
    """MLPBlock (tier 2, elementwise)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.gelu = nn.GELU()
        self.norm1.eval()
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = x + residual
        return x


INPUT_DIM = 256
HIDDEN_DIM = 512
BATCH_SIZE = 16
SEQ_LEN = 128

def get_inputs():
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    return [x]

def get_init_inputs():
    return [INPUT_DIM, HIDDEN_DIM]