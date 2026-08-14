import torch
import torch.nn as nn

class Model(nn.Module):
    """PoolElementwise (tier 3, pool)"""

    def __init__(self, pool_size, pool_stride, elementwise_lambda):
        super(Model, self).__init__()
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.elementwise_lambda = elementwise_lambda

    def forward(self, x):
        # Apply 2D average pooling
        x = torch.nn.functional.avg_pool2d(x, kernel_size=self.pool_size, stride=self.pool_stride)
        
        # Apply elementwise operation (scale by lambda)
        x = x * self.elementwise_lambda
        
        return x

# Module-level constants for shapes
BATCH_SIZE = 2
CHANNELS = 3
INPUT_HEIGHT = 16
INPUT_WIDTH = 16
POOL_SIZE = 2
POOL_STRIDE = 2
ELEMENTWISE_LAMBDA = 1.5

def get_inputs():
    """Generate input tensor for the model"""
    return [torch.randn(BATCH_SIZE, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)]

def get_init_inputs():
    """Return initialization arguments for the model"""
    return [POOL_SIZE, POOL_STRIDE, ELEMENTWISE_LAMBDA]