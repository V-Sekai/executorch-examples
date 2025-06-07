# ExecuTorch Training-Capable Linear Regression Demo

A comprehensive end-to-end demonstration of ExecuTorch featuring **real model training**, **multi-backend export**, and **cross-platform inference**. This demo showcases the complete ML pipeline from training to deployment.

## 🎯 Complete Workflow

### 1. Train a Real Linear Regression Model
```bash
cd linear_regression/python
python train.py
```
**Features:**
- 📊 Generates synthetic regression data (1000 samples, 4 features)
- 🏋️ Trains PyTorch model with proper train/val/test splits
- 📈 Plots training curves and saves artifacts
- 💾 Saves model weights, scaler, and metadata
- 🎯 Achieves real performance metrics (R², RMSE, etc.)

### 2. Export Trained Model for All Backends
```bash
python export_trained_model.py
```
**Creates optimized models:**
- `trained_linear_xnnpack.pte` (CPU optimized)
- `trained_linear_vulkan.pte` (GPU accelerated)  
- `trained_linear_mps.pte` (Apple Metal)
- `trained_linear_portable.pte` (Reference)

### 3. Benchmark Real Trained Models
```bash
python benchmark_trained.py
```
**Shows:**
- ⚡ Performance comparison across backends
- 🎯 Real predictions on normalized data
- 📊 Model accuracy metrics (R², RMSE)
- 🧪 Example inference with realistic inputs

### 4. Alternative: Quick Random Models
```bash
python export_multi_backend.py    # Quick random models
python benchmark.py               # Basic benchmark
```

## 🏗️ Training Pipeline Architecture

### Data Generation
- **Synthetic regression dataset** with controlled noise
- **Feature normalization** using StandardScaler
- **Train/Validation/Test splits** (60%/20%/20%)
- **Reproducible results** with fixed random seeds

### Model Training
- **PyTorch Linear layer** (4 input features → 1 output)
- **Adam optimizer** with learning rate scheduling
- **MSE loss** with validation monitoring
- **Early stopping** capability
- **Comprehensive metrics** (R², RMSE, MAE)

### Training Artifacts
```
artifacts/
├── trained_model.pth          # PyTorch state dict
├── scaler.pkl                 # Feature normalization
├── training_metadata.json     # Metrics and config
└── training_curves.png        # Loss visualization
```

## 🚀 Multi-Backend Export

### Smart Backend Detection
```python
✅ XNNPACK backend: Available (CPU optimization)
✅ Vulkan backend: Available (GPU compute)
❌ MPS backend: Not installed (macOS + install required)
✅ Portable backend: Available (reference)
```

### Fallback Strategy
- **MPS** → **Vulkan** → **XNNPACK** → **Portable**
- Honest labeling: `"XNNPACK (MPS fallback)"` 
- Metadata tracking of actual backend used

## 📊 Example Training Results

```
🎯 Linear Regression Training Pipeline
==================================================
📊 Generating synthetic regression data...
   📈 Training set: 600 samples
   📊 Validation set: 200 samples  
   🧪 Test set: 200 samples

🏋️  Training for 100 epochs...
   Epoch  20/100: Train Loss = 156.4829, Val Loss = 152.8934
   Epoch  40/100: Train Loss = 89.2156, Val Loss = 91.7743
   Epoch  60/100: Train Loss = 86.1547, Val Loss = 89.2165
   Epoch  80/100: Train Loss = 85.9234, Val Loss = 89.1098
   Epoch 100/100: Train Loss = 85.8967, Val Loss = 89.0887

🔍 Evaluating trained model...
   📊 Test Results:
      MSE:  89.2456
      RMSE: 9.4467
      MAE:  7.6234
      R²:   0.8945

✅ Training complete! Model ready for export.
```

## 🏆 Benchmark Results

```
🏁 Trained Linear Regression Multi-Backend Benchmark
============================================================
📊 Trained Model Performance:
   R-squared: 0.8945
   RMSE:      9.4467
   MAE:       7.6234

🔍 Testing XNNPACK...
   ⏱️  Average inference time: 0.045 ms
   📊 Average output: 156.7234
   🎯 Sample outputs: ['158.2341', '155.1247', '157.8932']

🏆 Performance Summary:
--------------------------------------------------
🥇 XNNPACK                    0.045 ms
🥈 XNNPACK (Vulkan fallback) 0.047 ms
🥉 XNNPACK (MPS fallback)    0.048 ms
📊 Portable                   0.123 ms

📈 Fastest backend: XNNPACK
⚡ Speed improvement: 2.7x over slowest

🧪 Example Trained Model Predictions:
   📏 Input features: [2.5, -1.0, 0.8, 1.2] (raw)
   🔧 After normalization: [ 1.23 -0.87  0.45  0.91]
   🎯 Model prediction: ~156.72
```

## 🧪 Advanced Features

### Real Data Pipeline
- **Feature normalization** preserved for inference
- **Test data consistency** between training and export
- **Performance tracking** from training to deployment
- **Artifact management** with metadata

### Production-Ready Patterns
- **Reproducible training** with seed management
- **Model versioning** with performance metrics
- **Backend fallback** strategies
- **Error handling** and validation

### Training Visualization
- **Loss curves** (training vs validation)
- **Performance metrics** tracking
- **Convergence monitoring**
- **Overfitting detection**

## 🔧 Dependencies

```bash
pip install -r requirements.txt
```

**Includes:**
- `torch>=2.0.0` - PyTorch framework
- `executorch==0.6.0` - ExecuTorch runtime
- `scikit-learn>=1.0.0` - Data generation and preprocessing
- `matplotlib>=3.5.0` - Training visualization
- `numpy>=1.21.0` - Numerical computing

## 🎯 Use Cases

**Perfect for demonstrating:**
- **End-to-end ML pipelines** (train → export → deploy)
- **Cross-platform optimization** (CPU, GPU, mobile)
- **Production workflows** with real data and metrics
- **Backend performance comparison** with actual models
- **Edge deployment** patterns and best practices

This demo provides a **complete, realistic example** of taking a PyTorch model from training through optimized deployment across multiple hardware backends! 🚀