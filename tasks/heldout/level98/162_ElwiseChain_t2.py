import torch
import torch.nn as nn

"""ElwiseChain (tier 2, elementwise)"""

# Module-level constants for shape
INPUT_FEATURES = 64
HIDDEN_FEATURES = 128
OUTPUT_FEATURES = 32
BATCH_SIZE = 4

class Model(nn.Module):
    """ElwiseChain (tier 2, elementwise)"""
    
    def __init__(self, input_features, hidden_features, output_features):
        super(Model, self).__init__()
        # Define linear layers for dimension transformation
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, output_features)
        
        # Use BatchNorm1d and set to eval mode for deterministic behavior
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.bn3 = nn.BatchNorm1d(output_features)
        self.bn1.eval()
        self.bn2.eval()
        self.bn3.eval()
    
    def forward(self, x):
        # Chain of elementwise operations
        # 1. Linear transformation
        out = self.fc1(x)
        # 2. Batch normalization
        out = self.bn1(out)
        # 3. Elementwise activation
        out = torch.sigmoid(out)
        # 4. Linear transformation
        out = self.fc2(out)
        # 5. Batch normalization
        out = self.bn2(out)
        # 6. Elementwise activation
        out = torch.relu(out)
        # 7. Linear transformation
        out = self.fc3(out)
        # 8. Batch normalization
        out = self.bn3(out)
        return out


def get_inputs():
    """Returns input tensors for the model."""
    return [torch.randn(BATCH_SIZE, INPUT_FEATURES)]


def get_init_inputs():
    """Returns initialization arguments for the model."""
    return [INPUT_FEATURES, HIDDEN_FEATURES, OUTPUT_FEATURES]