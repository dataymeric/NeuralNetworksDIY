import torch
import torch.nn as nn
from src.convolution import *

print("running tests for convolution...")

input = torch.randn(20, 3, 50) # N, C_IN, L_IN

# Conv1D
Conv = Conv1D(2, 3, 10, stride=2)
out_conv = Conv.forward(input.numpy().reshape((20, 50, 3)))
assert out_conv.shape == (20, 25, 10)

# MaxPool1D
MaxPool = MaxPool1D(2, 2)
out_maxpool = MaxPool.forward(input.numpy().reshape((20, 50, 3)))
assert out_maxpool.shape == (20, 25, 3)

# AvgPool1D
AvgPool = AvgPool1D(2, 2)
out_avgpool = AvgPool.forward(input.numpy().reshape((20, 50, 3)))
assert out_avgpool.shape == (20, 25, 3)

# Flatten
Flat = Flatten()
out_flat_fwd = Flat.forward(input.numpy().reshape((20, 50, 3)))
out_flat_bwd = Flat.backward(input.numpy().reshape((20, 50, 3)))

assert out_flat_fwd.shape == (20, 50 * 3)
assert out_flat_bwd.shape == (20, 50, 3)

print("tests passed ✅")