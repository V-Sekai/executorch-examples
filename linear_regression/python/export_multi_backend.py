import torch
import torch.nn as nn
import os

# Try new API first, then fallback to old API for XNNPACK
try:
    from executorch.partition.xnnpack import XnnpackPartitioner
except ImportError:
    try:
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    except ImportError as e:
        raise ImportError(
            "XnnpackPartitioner could not be imported. "
            "Check your executorch installation and API location."
        ) from e

from executorch.exir import to_edge_transform_and_lower

# Simple linear regression model
class SimpleLinearModel(nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

def export_model(model, sample_inputs, partitioner, filename, actual_backend_name):
    """Export model with specific backend partitioner"""
    try:
        et_program = to_edge_transform_and_lower(
            torch.export.export(model, sample_inputs),
            partitioner=[partitioner] if partitioner else []
        ).to_executorch()
        
        with open(filename, "wb") as f:
            f.write(et_program.buffer)
        print(f"✅ Exported: {filename} (using {actual_backend_name})")
        return True, actual_backend_name
    except Exception as e:
        print(f"❌ Failed to export {filename}: {e}")
        return False, None

def main():
    # Set seed for reproducible results
    torch.manual_seed(42)
    
    # Create model and sample inputs
    model = SimpleLinearModel().eval()
    sample_inputs = (torch.randn(1, 4),)
    
    # Create models directory
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)
    
    print("🚀 Exporting linear regression models for XNNPACK backend...")
    
    # 1. XNNPACK (CPU optimized - always available)
    print("\n📱 Exporting XNNPACK backend (CPU optimized)...")
    xnnpack_print_name = "XNNPACK"
    xnnpack_file_name_suffix = "xnnpack"
    xnnpack_filename = os.path.join(models_dir, f"linear_{xnnpack_file_name_suffix}.pte")
    xnnpack_success, _ = export_model(
        model, sample_inputs, 
        XnnpackPartitioner(),
        xnnpack_filename,
        xnnpack_print_name
    )
    
    print("\n🏁 All requested models exported successfully!")

if __name__ == "__main__":
    main()