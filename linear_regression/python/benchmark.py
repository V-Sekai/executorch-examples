import torch
import time
import os
import json
from executorch.runtime import Runtime

def benchmark_model(model_path, model_name, iterations=100):
    """Benchmark a model and return performance metrics"""
    if not os.path.exists(model_path):
        return None
    
    try:
        runtime = Runtime.get()
        program = runtime.load_program(model_path)
        method = program.load_method("forward")
        
        # Use fixed input for consistent comparison
        torch.manual_seed(42)
        input_tensor = torch.randn(1, 4)
        
        # Warmup
        for _ in range(10):
            method.execute([input_tensor])
        
        # Benchmark
        start_time = time.time()
        for _ in range(iterations):
            output = method.execute([input_tensor])
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / iterations
        
        return {
            'name': model_name,
            'avg_time_ms': avg_time_ms,
            'output': output[0].item() if output else None
        }
    except Exception as e:
        print(f"❌ Error benchmarking {model_name}: {e}")
        return None

def main():
    print("🏁 Linear Regression Multi-Backend Benchmark")
    print("=" * 50)
    
    models_dir = "../models"
    
    # Try to read actual backend info from metadata
    metadata_file = os.path.join(models_dir, "backend_metadata.json")
    backend_metadata = {}
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            backend_metadata = json.load(f)
    
    models_to_test = [
        ("linear_xnnpack.pte", backend_metadata.get("linear_xnnpack.pte", "XNNPACK (CPU)")),
        ("linear_vulkan.pte", backend_metadata.get("linear_vulkan.pte", "Vulkan/XNNPACK")),  
        ("linear_mps.pte", backend_metadata.get("linear_mps.pte", "MPS/XNNPACK")),
        ("linear_portable.pte", backend_metadata.get("linear_portable.pte", "Portable"))
    ]
    
    results = []
    
    for model_file, model_name in models_to_test:
        model_path = os.path.join(models_dir, model_file)
        print(f"\n🔍 Testing {model_name}...")
        
        result = benchmark_model(model_path, model_name)
        if result:
            results.append(result)
            print(f"   ⏱️  Average inference time: {result['avg_time_ms']:.3f} ms")
            print(f"   📊 Output value: {result['output']:.6f}")
        else:
            print(f"   ❌ Model not available or failed")
    
    # Summary
    if results:
        print("\n🏆 Performance Summary:")
        print("-" * 50)
        
        # Check if all outputs are the same (within tolerance)
        outputs = [r['output'] for r in results if r['output'] is not None]
        if outputs and all(abs(outputs[0] - output) < 0.001 for output in outputs):
            print("✅ All models produce identical outputs (same weights)")
        else:
            print("⚠️  Models produce different outputs (different weights)")
        
        # Sort by performance
        results.sort(key=lambda x: x['avg_time_ms'])
        
        for i, result in enumerate(results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            print(f"{rank} {result['name']:<20} {result['avg_time_ms']:>8.3f} ms")
        
        print(f"\n📈 Fastest backend: {results[0]['name']}")
        if len(results) > 1:
            print(f"⚡ Speed improvement: {results[-1]['avg_time_ms'] / results[0]['avg_time_ms']:.1f}x")
    else:
        print("\n❌ No models could be benchmarked")

if __name__ == "__main__":
    main()