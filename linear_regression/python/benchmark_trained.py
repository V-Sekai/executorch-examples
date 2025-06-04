import torch
import time
import os
import json
import pickle
import numpy as np
from executorch.runtime import Runtime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def load_test_data():
    """Load test data and scalers from training artifacts"""
    artifacts_dir = "../artifacts"
    
    # Load scalers
    scaler_X_path = os.path.join(artifacts_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(artifacts_dir, "scaler_y.pkl")
    
    scalers = {}
    if os.path.exists(scaler_X_path):
        with open(scaler_X_path, "rb") as f:
            scalers['X'] = pickle.load(f)
    if os.path.exists(scaler_y_path):
        with open(scaler_y_path, "rb") as f:
            scalers['y'] = pickle.load(f)
    
    return scalers if scalers else None

def get_analytical_solution(scalers):
    """Get the analytical least squares solution from the training data"""
    artifacts_dir = "../artifacts"
    metadata_path = os.path.join(artifacts_dir, "training_metadata.json")
    
    if not os.path.exists(metadata_path):
        return None
    
    # Regenerate the same training data used in training
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Use same parameters as training
    np.random.seed(42)
    X, y = make_regression(
        n_samples=2000,
        n_features=4,
        noise=5.0,
        random_state=42,
        bias=50.0,
        n_informative=4,
        effective_rank=None,
        tail_strength=0.5
    )
    
    # Apply same preprocessing
    if scalers and 'X' in scalers and 'y' in scalers:
        X_normalized = scalers['X'].transform(X)
        y_normalized = scalers['y'].transform(y.reshape(-1, 1)).flatten()
    else:
        X_normalized = X
        y_normalized = y
    
    # Split the same way as training
    X_train, _, y_train, _ = train_test_split(X_normalized, y_normalized, test_size=0.4, random_state=42)
    
    # Compute analytical solution: w = (X^T X)^(-1) X^T y
    X_train_with_bias = np.column_stack([X_train, np.ones(X_train.shape[0])])
    
    try:
        # Normal equation solution
        XtX = X_train_with_bias.T @ X_train_with_bias
        Xty = X_train_with_bias.T @ y_train
        analytical_params = np.linalg.solve(XtX, Xty)
        
        weights = analytical_params[:-1]
        bias = analytical_params[-1]
        
        return {
            'weights': weights,
            'bias': bias,
            'method': 'Normal Equation'
        }
    except np.linalg.LinAlgError:
        # Fallback to SVD if matrix is singular
        try:
            analytical_params = np.linalg.lstsq(X_train_with_bias, y_train, rcond=None)[0]
            weights = analytical_params[:-1]
            bias = analytical_params[-1]
            
            return {
                'weights': weights,
                'bias': bias,
                'method': 'SVD (Least Squares)'
            }
        except:
            return None

def predict_analytical(weights, bias, X):
    """Make predictions using analytical solution"""
    return X @ weights + bias

def compare_with_sklearn(scalers):
    """Compare with sklearn's LinearRegression for validation"""
    artifacts_dir = "../artifacts"
    
    # Regenerate training data
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    np.random.seed(42)
    X, y = make_regression(
        n_samples=2000,
        n_features=4,
        noise=5.0,
        random_state=42,
        bias=50.0,
        n_informative=4,
        effective_rank=None,
        tail_strength=0.5
    )
    
    # Apply same preprocessing
    if scalers and 'X' in scalers and 'y' in scalers:
        X_normalized = scalers['X'].transform(X)
        y_normalized = scalers['y'].transform(y.reshape(-1, 1)).flatten()
    else:
        X_normalized = X
        y_normalized = y
    
    # Split the same way
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.4, random_state=42)
    
    # Train sklearn model
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = sklearn_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'weights': sklearn_model.coef_,
        'bias': sklearn_model.intercept_,
        'r2': r2,
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse)
    }

def benchmark_trained_model(model_path, model_name, scalers=None, iterations=100):
    """Benchmark a trained model with realistic test cases"""
    if not os.path.exists(model_path):
        return None
    
    try:
        runtime = Runtime.get()
        program = runtime.load_program(model_path)
        method = program.load_method("forward")
        
        # Create realistic test inputs
        torch.manual_seed(42)
        
        # Test case 1: Normalized random features
        test_input_1 = torch.randn(1, 4)
        
        # Test case 2: Specific feature values (if we have scalers)
        if scalers and 'X' in scalers:
            # Create some realistic raw features and normalize them
            raw_features = np.array([[2.5, -1.0, 0.8, 1.2]])  # Example real-world values
            normalized_features = scalers['X'].transform(raw_features)
            test_input_2 = torch.FloatTensor(normalized_features)
        else:
            test_input_2 = torch.randn(1, 4)
        
        # Warmup
        for _ in range(10):
            method.execute([test_input_1])
        
        # Benchmark
        start_time = time.time()
        outputs = []
        for i in range(iterations):
            test_input = test_input_1 if i % 2 == 0 else test_input_2
            output = method.execute([test_input])
            outputs.append(output[0].item())
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / iterations
        
        # Denormalize outputs if we have the y scaler
        if scalers and 'y' in scalers:
            denormalized_outputs = [
                scalers['y'].inverse_transform([[out]])[0, 0] 
                for out in outputs[:5]
            ]
            avg_denormalized = scalers['y'].inverse_transform([[np.mean(outputs)]])[0, 0]
        else:
            denormalized_outputs = outputs[:5]
            avg_denormalized = np.mean(outputs)
        
        return {
            'name': model_name,
            'avg_time_ms': avg_time_ms,
            'sample_outputs': outputs[:5],
            'sample_outputs_real': denormalized_outputs,
            'avg_output': np.mean(outputs),
            'avg_output_real': avg_denormalized,
            'output_std': np.std(outputs)
        }
    except Exception as e:
        print(f"❌ Error benchmarking {model_name}: {e}")
        return None

def main():
    print("🏁 Trained Linear Regression Multi-Backend Benchmark")
    print("=" * 60)
    
    models_dir = "../models"
    
    # Check if trained models exist
    trained_metadata_file = os.path.join(models_dir, "trained_backend_metadata.json")
    if not os.path.exists(trained_metadata_file):
        print("❌ No trained models found!")
        print("🏋️  Please run:")
        print("   1. python train.py")
        print("   2. python export_trained_model.py")
        return
    
    # Load trained model metadata
    with open(trained_metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load test data scalers
    scalers = load_test_data()
    if scalers:
        print("✅ Loaded data scalers for realistic test inputs")
    else:
        print("⚠️  No scalers found, using random test inputs")
    
    # Get analytical solution for comparison
    print("\n🧮 Computing Analytical Solution...")
    analytical = get_analytical_solution(scalers)
    sklearn_results = compare_with_sklearn(scalers)
    
    if analytical and sklearn_results:
        print(f"✅ Analytical solution computed using {analytical['method']}")
        print(f"   📊 Sklearn validation: R² = {sklearn_results['r2']:.4f}, RMSE = {sklearn_results['rmse']:.4f}")
        print(f"   🎯 Analytical weights: {analytical['weights']}")
        print(f"   🎯 Analytical bias: {analytical['bias']:.4f}")
        print(f"   🔧 Sklearn weights: {sklearn_results['weights']}")
        print(f"   🔧 Sklearn bias: {sklearn_results['bias']:.4f}")
        
        # Check if they match
        weights_match = np.allclose(analytical['weights'], sklearn_results['weights'], atol=1e-4)
        bias_match = np.allclose(analytical['bias'], sklearn_results['bias'], atol=1e-4)
        
        if weights_match and bias_match:
            print("   ✅ Analytical and sklearn solutions match perfectly!")
        else:
            print("   ⚠️  Small differences between analytical and sklearn (numerical precision)")
    else:
        print("❌ Could not compute analytical solution")
        analytical = None
        sklearn_results = None
    
    # Display model performance
    if 'model_performance' in metadata:
        perf = metadata['model_performance']
        print(f"\n📊 Trained Model Performance:")
        print(f"   R-squared: {perf['r2']:.4f}")
        print(f"   RMSE:      {perf['rmse']:.4f}")
        print(f"   MAE:       {perf['mae']:.4f}")
    
    models_to_test = [
        ("trained_linear_xnnpack.pte", metadata.get("trained_linear_xnnpack.pte", "XNNPACK")),
        ("trained_linear_vulkan.pte", metadata.get("trained_linear_vulkan.pte", "Vulkan")),  
        ("trained_linear_mps.pte", metadata.get("trained_linear_mps.pte", "MPS")),
        ("trained_linear_portable.pte", metadata.get("trained_linear_portable.pte", "Portable"))
    ]
    
    results = []
    
    for model_file, model_name in models_to_test:
        model_path = os.path.join(models_dir, model_file)
        print(f"\n🔍 Testing {model_name}...")
        
        result = benchmark_trained_model(model_path, model_name, scalers)
        if result:
            results.append(result)
            print(f"   ⏱️  Average inference time: {result['avg_time_ms']:.3f} ms")
            print(f"   📊 Average output (normalized): {result['avg_output']:.4f}")
            if 'avg_output_real' in result:
                print(f"   🎯 Average output (real scale): {result['avg_output_real']:.4f}")
            print(f"   📈 Output std dev: {result['output_std']:.4f}")
            if 'sample_outputs_real' in result:
                print(f"   🔍 Sample outputs (real): {[f'{x:.2f}' for x in result['sample_outputs_real'][:3]]}")
            else:
                print(f"   🔍 Sample outputs: {[f'{x:.4f}' for x in result['sample_outputs'][:3]]}")
        else:
            print(f"   ❌ Model not available or failed")
    
    # Summary
    if results:
        print("\n🏆 Performance Summary:")
        print("-" * 60)
        
        # Check output consistency
        avg_outputs = [r['avg_output'] for r in results]
        if avg_outputs and all(abs(avg_outputs[0] - output) < 0.1 for output in avg_outputs):
            print("✅ All models produce consistent outputs (same trained weights)")
        else:
            print("⚠️  Models produce different outputs")
        
        # Sort by performance
        results.sort(key=lambda x: x['avg_time_ms'])
        
        for i, result in enumerate(results):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "📊"
            print(f"{rank} {result['name']:<25} {result['avg_time_ms']:>8.3f} ms")
        
        print(f"\n📈 Fastest backend: {results[0]['name']}")
        if len(results) > 1:
            speedup = results[-1]['avg_time_ms'] / results[0]['avg_time_ms']
            print(f"⚡ Speed improvement: {speedup:.1f}x over slowest")
        
        # Show realistic prediction example
        print(f"\n🧪 Example Trained Model Predictions:")
        if scalers and 'X' in scalers:
            raw_example = [2.5, -1.0, 0.8, 1.2]
            normalized_example = scalers['X'].transform([raw_example])[0]
            print(f"   📏 Input features: {raw_example} (raw)")
            print(f"   🔧 After normalization: {normalized_example}")
            
            # Compare model vs analytical predictions
            if analytical:
                analytical_pred = predict_analytical(analytical['weights'], analytical['bias'], normalized_example.reshape(1, -1))[0]
                if scalers and 'y' in scalers:
                    analytical_pred_real = scalers['y'].inverse_transform([[analytical_pred]])[0, 0]
                    print(f"   🎯 Analytical prediction: {analytical_pred_real:.2f} (real scale)")
                else:
                    print(f"   🎯 Analytical prediction: {analytical_pred:.2f}")
            
            if 'avg_output_real' in results[0]:
                print(f"   🤖 Model prediction: ~{results[0]['avg_output_real']:.2f} (real scale)")
                
                # Compare accuracy
                if analytical and 'avg_output_real' in results[0]:
                    if scalers and 'y' in scalers:
                        diff = abs(analytical_pred_real - results[0]['avg_output_real'])
                        if diff < 1.0:
                            print(f"   ✅ Model vs Analytical: Difference = {diff:.2f} (excellent match!)")
                        elif diff < 5.0:
                            print(f"   👍 Model vs Analytical: Difference = {diff:.2f} (good match)")
                        else:
                            print(f"   ⚠️  Model vs Analytical: Difference = {diff:.2f} (check training)")
            else:
                print(f"   🤖 Model prediction: ~{results[0]['avg_output']:.2f} (normalized)")
        else:
            print(f"   🎯 Model prediction: ~{results[0]['avg_output']:.2f}")
        
        # Load actual trained model weights for comparison
        artifacts_dir = "../artifacts"
        model_path = os.path.join(artifacts_dir, "trained_model.pth")
        if os.path.exists(model_path) and analytical:
            print(f"\n📊 Weight Comparison:")
            trained_weights = torch.load(model_path, map_location='cpu')
            pytorch_weights = trained_weights['linear.weight'].numpy().flatten()
            pytorch_bias = trained_weights['linear.bias'].numpy()[0]
            
            print(f"   🧮 Analytical weights: {analytical['weights']}")
            print(f"   🤖 PyTorch weights:    {pytorch_weights}")
            print(f"   🧮 Analytical bias:    {analytical['bias']:.6f}")
            print(f"   🤖 PyTorch bias:       {pytorch_bias:.6f}")
            
            # Compare weights
            weight_diff = np.linalg.norm(analytical['weights'] - pytorch_weights)
            bias_diff = abs(analytical['bias'] - pytorch_bias)
            
            print(f"   📏 Weight difference (L2 norm): {weight_diff:.6f}")
            print(f"   📏 Bias difference:             {bias_diff:.6f}")
            
            if weight_diff < 0.01 and bias_diff < 0.01:
                print(f"   ✅ PyTorch training converged to optimal solution!")
            elif weight_diff < 0.1 and bias_diff < 0.1:
                print(f"   👍 PyTorch training very close to optimal solution")
            else:
                print(f"   ⚠️  PyTorch weights differ from analytical (training/optimization effects)")
        
    else:
        print("\n❌ No models could be benchmarked")

if __name__ == "__main__":
    main()