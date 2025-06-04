import torch
import torch.nn as nn
import os
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

# Simple linear regression model
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleLinearModel().eval()
sample_inputs = (torch.randn(1, 4), )

et_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()]
).to_executorch()

# Create models directory in linear_regression folder
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)

with open(os.path.join(models_dir, "simple_linear.pte"), "wb") as f:
    f.write(et_program.buffer)
