import torch
import torch.nn as nn
import os
import platform
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
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
    
    print("🚀 Exporting linear regression models for multiple backends...")
    
    # 1. XNNPACK (CPU optimized - always available)
    print("\n📱 Exporting XNNPACK backend (CPU optimized)...")
    xnnpack_success, xnnpack_backend = export_model(
        model, sample_inputs, 
        XnnpackPartitioner(),
        os.path.join(models_dir, "linear_xnnpack.pte"),
        "XNNPACK"
    )
    
    # 2. Vulkan (GPU compute - if available)
    print("\n🎮 Exporting Vulkan backend (GPU compute)...")
    try:
        from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
        vulkan_success, vulkan_backend = export_model(
            model, sample_inputs,
            VulkanPartitioner(),
            os.path.join(models_dir, "linear_vulkan.pte"),
            "Vulkan"
        )
    except ImportError:
        print("⚠️  Vulkan backend not available, creating fallback to XNNPACK...")
        vulkan_success, vulkan_backend = export_model(
            model, sample_inputs,
            XnnpackPartitioner(),
            os.path.join(models_dir, "linear_vulkan.pte"),
            "XNNPACK (Vulkan fallback)"
        )
    
    # 3. MPS (Apple Metal - if on macOS)
    print("\n🍎 Exporting MPS backend (Apple Metal)...")
    if platform.system() == "Darwin":
        try:
            from executorch.backends.mps.partition.mps_partitioner import MPSPartitioner
            mps_success, mps_backend = export_model(
                model, sample_inputs,
                MPSPartitioner(),
                os.path.join(models_dir, "linear_mps.pte"),
                "MPS"
            )
        except ImportError:
            print("⚠️  MPS backend not available, creating fallback to XNNPACK...")
            mps_success, mps_backend = export_model(
                model, sample_inputs,
                XnnpackPartitioner(),
                os.path.join(models_dir, "linear_mps.pte"),
                "XNNPACK (MPS fallback)"
            )
    else:
        print("⚠️  Not on macOS, creating fallback to XNNPACK...")
        mps_success, mps_backend = export_model(
            model, sample_inputs,
            XnnpackPartitioner(),
            os.path.join(models_dir, "linear_mps.pte"),
            "XNNPACK (macOS fallback)"
        )
    
    # 4. Portable (no backend optimization - reference)
    print("\n🔧 Exporting Portable backend (reference implementation)...")
    portable_success, portable_backend = export_model(
        model, sample_inputs,
        None,  # No partitioner = portable
        os.path.join(models_dir, "linear_portable.pte"),
        "Portable"
    )
    
    # Summary
    print("\n📊 Export Summary:")
    print(f"   XNNPACK (CPU): {'✅' if xnnpack_success else '❌'} ({xnnpack_backend})")
    print(f"   Vulkan (GPU):  {'✅' if vulkan_success else '❌'} ({vulkan_backend})")
    print(f"   MPS (Apple):   {'✅' if mps_success else '❌'} ({mps_backend})")
    print(f"   Portable:      {'✅' if portable_success else '❌'} ({portable_backend})")
    
    # Save backend metadata for benchmark script
    metadata = {
        'linear_xnnpack.pte': xnnpack_backend,
        'linear_vulkan.pte': vulkan_backend, 
        'linear_mps.pte': mps_backend,
        'linear_portable.pte': portable_backend
    }
    
    import json
    with open(os.path.join(models_dir, "backend_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    if all([xnnpack_success, vulkan_success, mps_success, portable_success]):
        print("\n🎉 All models exported successfully!")
        print("📁 Models available in:", os.path.abspath(models_dir))
    else:
        print("\n⚠️  Some exports failed, but demo can still run with available backends.")

if __name__ == "__main__":
    main()