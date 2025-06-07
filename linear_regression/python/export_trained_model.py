import torch
import torch.nn as nn
import os
import platform
import pickle
import json
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

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size=4, output_size=1):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        return self.linear(x)

def load_trained_model():
    """Load the trained model from artifacts"""
    artifacts_dir = "../artifacts"
    
    # Load metadata
    metadata_path = os.path.join(artifacts_dir, "training_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("No trained model found. Run 'python train.py' first.")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Initialize model with same architecture
    model_arch = metadata['model_architecture']
    model = LinearRegressionModel(
        input_size=model_arch['input_size'],
        output_size=model_arch['output_size']
    )
    
    # Load trained weights
    model_path = os.path.join(artifacts_dir, "trained_model.pth")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"✅ Loaded trained model (R² = {metadata['training_metrics']['r2']:.4f})")
    return model, metadata

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
    print("🚀 Exporting Trained Linear Regression Model")
    print("=" * 50)
    
    # Load trained model
    try:
        model, metadata = load_trained_model()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("🏋️  Please run 'python train.py' first to train a model.")
        return
    
    # Create models directory
    models_dir = "../models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Use same sample inputs as training (normalized)
    torch.manual_seed(42)
    sample_inputs = (torch.randn(1, 4),)  # Same as training normalization
    
    print(f"\n📊 Model Performance Summary:")
    metrics = metadata['training_metrics']
    print(f"   MSE:  {metrics['mse']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   R²:   {metrics['r2']:.4f}")
    
    print(f"\n🎯 Exporting trained model for multiple backends...")
    
    # Check backend availability
    backends_available = {
        'xnnpack': True,
        'vulkan': False,
        'mps': False,
        'portable': True
    }
    
    try:
        from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
        backends_available['vulkan'] = True
        print("✅ Vulkan backend: Available")
    except ImportError:
        print("❌ Vulkan backend: Not installed")
    
    if platform.system() == "Darwin":
        try:
            from executorch.backends.mps.partition.mps_partitioner import MPSPartitioner
            backends_available['mps'] = True
            print("✅ MPS backend: Available")
        except ImportError:
            print("❌ MPS backend: Not installed")
    else:
        print("ℹ️  MPS backend: Not available (macOS only)")
    
    print("✅ XNNPACK backend: Available")
    print("✅ Portable backend: Available")
    
    # 1. XNNPACK (CPU optimized)
    print("\n📱 Exporting XNNPACK backend...")
    xnnpack_success, xnnpack_backend = export_model(
        model, sample_inputs, 
        XnnpackPartitioner(),
        os.path.join(models_dir, "trained_linear_xnnpack.pte"),
        "XNNPACK"
    )
    
    # 2. Vulkan (GPU compute)
    print("\n🎮 Exporting Vulkan backend...")
    if backends_available['vulkan']:
        from executorch.backends.vulkan.partition.vulkan_partitioner import VulkanPartitioner
        vulkan_success, vulkan_backend = export_model(
            model, sample_inputs,
            VulkanPartitioner(),
            os.path.join(models_dir, "trained_linear_vulkan.pte"),
            "Vulkan"
        )
    else:
        print("⚠️  Vulkan backend not available, creating fallback...")
        vulkan_success, vulkan_backend = export_model(
            model, sample_inputs,
            XnnpackPartitioner(),
            os.path.join(models_dir, "trained_linear_vulkan.pte"),
            "XNNPACK (Vulkan fallback)"
        )
    
    # 3. MPS (Apple Metal)
    print("\n🍎 Exporting MPS backend...")
    if backends_available['mps']:
        from executorch.backends.mps.partition.mps_partitioner import MPSPartitioner
        mps_success, mps_backend = export_model(
            model, sample_inputs,
            MPSPartitioner(),
            os.path.join(models_dir, "trained_linear_mps.pte"),
            "MPS"
        )
    else:
        if platform.system() == "Darwin":
            print("⚠️  MPS backend not available, creating fallback...")
        else:
            print("⚠️  Not on macOS, creating fallback...")
        mps_success, mps_backend = export_model(
            model, sample_inputs,
            XnnpackPartitioner(),
            os.path.join(models_dir, "trained_linear_mps.pte"),
            "XNNPACK (MPS fallback)"
        )
    
    # 4. Portable (reference)
    print("\n🔧 Exporting Portable backend...")
    portable_success, portable_backend = export_model(
        model, sample_inputs,
        None,
        os.path.join(models_dir, "trained_linear_portable.pte"),
        "Portable"
    )
    
    # Save metadata
    export_metadata = {
        'trained_linear_xnnpack.pte': xnnpack_backend,
        'trained_linear_vulkan.pte': vulkan_backend, 
        'trained_linear_mps.pte': mps_backend,
        'trained_linear_portable.pte': portable_backend,
        'model_performance': metrics,
        'training_completed': True
    }
    
    with open(os.path.join(models_dir, "trained_backend_metadata.json"), "w") as f:
        json.dump(export_metadata, f, indent=2)
    
    # Summary
    print(f"\n📊 Export Summary:")
    print(f"   XNNPACK (CPU): {'✅' if xnnpack_success else '❌'} ({xnnpack_backend})")
    print(f"   Vulkan (GPU):  {'✅' if vulkan_success else '❌'} ({vulkan_backend})")
    print(f"   MPS (Apple):   {'✅' if mps_success else '❌'} ({mps_backend})")
    print(f"   Portable:      {'✅' if portable_success else '❌'} ({portable_backend})")
    
    if all([xnnpack_success, vulkan_success, mps_success, portable_success]):
        print(f"\n🎉 All trained models exported successfully!")
        print(f"📁 Models available in: {os.path.abspath(models_dir)}")
        print(f"📈 Model performance: R² = {metrics['r2']:.4f}")
        print(f"\n🧪 Run 'python benchmark_trained.py' to test the trained models!")
    else:
        print(f"\n⚠️  Some exports failed, but demo can still run with available backends.")

if __name__ == "__main__":
    main()