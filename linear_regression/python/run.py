import torch
import torch.nn as nn
from executorch.runtime import Runtime
from typing import List

runtime = Runtime.get()

input_tensor: torch.Tensor = torch.randn(1, 4)
program = runtime.load_program("simple_linear.pte")
method = program.load_method("forward")
output: List[torch.Tensor] = method.execute([input_tensor])
print("✅ Linear regression model executed successfully via ExecuTorch!")
print(f"📊 Input features: {input_tensor.numpy().flatten()}")
print(f"🎯 Predicted output: {output[0].item():.4f}")

class SimpleLinearModel(nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

eager_reference_model = SimpleLinearModel().eval()
eager_reference_output = eager_reference_model(input_tensor)

print("\n🔬 Model Validation:")
print("Comparing ExecuTorch vs PyTorch outputs...")
is_close = torch.allclose(output[0], eager_reference_output, rtol=1e-3, atol=1e-5)
print(f"✅ Models match within tolerance: {is_close}")
print(f"📈 ExecuTorch prediction: {output[0].item():.6f}")
print(f"📈 PyTorch prediction:   {eager_reference_output.item():.6f}")
print(f"📊 Absolute difference:  {abs(output[0].item() - eager_reference_output.item()):.6f}")

if is_close:
    print("\n🎉 SUCCESS: Linear regression model conversion and execution verified!")
else:
    print("\n⚠️  WARNING: Output mismatch detected!")
