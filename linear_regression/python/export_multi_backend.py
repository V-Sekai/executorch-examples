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
    xnnpack_print_name = "XNNPACK"
    xnnpack_file_name_suffix = "xnnpack"
    xnnpack_filename = os.path.join(models_dir, f"linear_{xnnpack_file_name_suffix}.pte")
    xnnpack_success, _ = export_model(
        model, sample_inputs, 
        XnnpackPartitioner(),
        xnnpack_filename,
        xnnpack_print_name
    )
    
    # 2. Vulkan (GPU compute - if available)
    print("\n🎮 Exporting Vulkan backend (GPU compute)...")
    vulkan_print_name = "Vulkan" # Default print name
    vulkan_file_name_suffix = "vulkan" # Default file suffix
    try:
        from executorch.partition.vulkan import VulkanPartitioner
        vulkan_filename = os.path.join(models_dir, f"linear_{vulkan_file_name_suffix}.pte")
        vulkan_success, _ = export_model(
            model, sample_inputs,
            VulkanPartitioner(),
            vulkan_filename,
            vulkan_print_name
        )
    except ImportError:
        print("⚠️  Vulkan backend not available, creating fallback to XNNPACK...")
        vulkan_print_name = "XNNPACK (Vulkan fallback)"
        vulkan_file_name_suffix = "xnnpack_as_vulkan_fallback"
        vulkan_filename = os.path.join(models_dir, f"linear_{vulkan_file_name_suffix}.pte")
        vulkan_success, _ = export_model(
            model, sample_inputs,
            XnnpackPartitioner(),
            vulkan_filename,
            vulkan_print_name
        )
    
    # 3. MPS (Apple Metal - if on macOS)
    print("\n🍎 Exporting MPS backend (Apple Metal)...")
    mps_print_name = "MPS" # Default print name
    mps_file_name_suffix = "mps" # Default file suffix
    if platform.system() == "Darwin":
        try:
            from executorch.partition.mps import MPSPartitioner
            mps_filename = os.path.join(models_dir, f"linear_{mps_file_name_suffix}.pte")
            mps_success, _ = export_model(
                model, sample_inputs,
                MPSPartitioner(),
                mps_filename,
                mps_print_name
            )
        except ImportError:
            print("⚠️  MPS backend not available, creating fallback to XNNPACK...")
            mps_print_name = "XNNPACK (MPS fallback)"
            mps_file_name_suffix = "xnnpack_as_mps_fallback"
            mps_filename = os.path.join(models_dir, f"linear_{mps_file_name_suffix}.pte")
            mps_success, _ = export_model(
                model, sample_inputs,
                XnnpackPartitioner(),
                mps_filename,
                mps_print_name
            )
    else:
        print("⚠️  Not on macOS, creating fallback to XNNPACK...")
        mps_print_name = "XNNPACK (macOS fallback)"
        mps_file_name_suffix = "xnnpack_as_macos_fallback"
        mps_filename = os.path.join(models_dir, f"linear_{mps_file_name_suffix}.pte")
        mps_success, _ = export_model(
            model, sample_inputs,
            XnnpackPartitioner(),
            mps_filename,
            mps_print_name
        )
    
    # 4. Portable (no backend optimization - reference)
    print("\n🔧 Exporting Portable backend (reference implementation)...")
    portable_print_name = "Portable"
    portable_file_name_suffix = "portable"
    portable_filename = os.path.join(models_dir, f"linear_{portable_file_name_suffix}.pte")
    portable_success, _ = export_model(
        model, sample_inputs,
        None,  # No partitioner = portable
        portable_filename,
        portable_print_name
    )
    
    # Summary
    print("\n📊 Export Summary:")
    print(f"   XNNPACK (CPU): {'✅' if xnnpack_success else '❌'} (linear_{xnnpack_file_name_suffix}.pte)")
    print(f"   Vulkan (GPU):  {'✅' if vulkan_success else '❌'} (linear_{vulkan_file_name_suffix}.pte)")
    print(f"   MPS (Apple):   {'✅' if mps_success else '❌'} (linear_{mps_file_name_suffix}.pte)")
    print(f"   Portable:      {'✅' if portable_success else '❌'} (linear_{portable_file_name_suffix}.pte)")
    
    if all([xnnpack_success, vulkan_success, mps_success, portable_success]):
        print("\n🎉 All models exported successfully!")
        print("📁 Models available in:", os.path.abspath(models_dir))
    else:
        print("\n⚠️  Some exports failed, but demo can still run with available backends.")

if __name__ == "__main__":
    main()