# ExecuTorch Multi-Backend Linear Regression Demo

A comprehensive demonstration of ExecuTorch's cross-platform capabilities, showcasing linear regression inference across multiple backends: **XNNPACK** (CPU), **Vulkan** (GPU), and **MPS** (Apple Metal).

## 🚀 Quick Start

### 1. Export Models for All Backends
```bash
cd linear_regression/python
python export_multi_backend.py
```

### 2. Run Python Benchmark
```bash
python benchmark.py
```

### 3. Build and Run C++ Demo
```bash
cd ../cpp
chmod +x build_multi.sh
./build_multi.sh
./../../build_multi/bin/executorch_multi_backend_demo
```

## 🏗️ Architecture

### Backend Support
- **🍎 MPS (Apple Metal)**: Hardware-accelerated inference on Apple Silicon
- **🎮 Vulkan**: Cross-platform GPU compute acceleration  
- **📱 XNNPACK**: Highly optimized CPU kernels
- **🔧 Portable**: Reference implementation (no optimization)

### Smart Fallback System
The demo automatically detects available hardware and falls back gracefully:
```
MPS (Apple) → Vulkan (GPU) → XNNPACK (CPU) → Portable
```

## 📊 Features

### Python Tools
- **`export_multi_backend.py`**: Exports models for all supported backends
- **`benchmark.py`**: Performance comparison across backends
- Automatic platform detection and fallback model generation

### C++ Demo Application  
- **Multi-backend testing**: Automatically tests all available backends
- **Performance benchmarking**: 1000-iteration timing with warmup
- **Intelligent ranking**: Shows performance hierarchy
- **Cross-platform compatibility**: Builds on Linux, macOS, and Windows

### Example Output
```
🚀 ExecuTorch Multi-Backend Linear Regression Demo
=======================================================
📊 Input features: [1, 2, 3, 4]

✅ MPS (Apple Metal): 0.012 ms avg
✅ Vulkan (GPU): 0.018 ms avg  
✅ XNNPACK (CPU): 0.045 ms avg
✅ Portable (Reference): 0.123 ms avg

🏆 Performance Ranking:
--------------------------------------------------
🥇 MPS (Apple Metal)         0.012 ms (output: 2.718281)
🥈 Vulkan (GPU)              0.018 ms (output: 2.718281) 
🥉 XNNPACK (CPU)             0.045 ms (output: 2.718281)
📊 Portable (Reference)      0.123 ms (output: 2.718281)

📈 Summary:
   🏃 Fastest: MPS (Apple Metal)
   ⚡ Speed improvement: 10.2x over slowest

✅ ExecuTorch multi-backend demo completed successfully!
```

## 🔧 Dependencies

### Python
- PyTorch 2.0+
- ExecuTorch 0.6.0
- Platform-specific backend packages (auto-detected)

### C++
- CMake 3.18+
- C++17 compatible compiler
- Vulkan SDK (optional, for GPU acceleration)

## 🧪 CI/CD Testing

GitHub Actions workflow tests across:
- **Ubuntu**: XNNPACK + Vulkan
- **macOS**: MPS + XNNPACK  
- **Windows**: XNNPACK

## 🎯 Use Cases

This demo showcases ExecuTorch's ability to:
- **Optimize for target hardware**: Automatically select the fastest backend
- **Provide consistent APIs**: Same inference code across all backends
- **Ensure portability**: Graceful fallback when specialized hardware unavailable
- **Deliver performance**: Up to 10x speedup with hardware acceleration

Perfect for applications requiring:
- Cross-platform deployment
- Hardware-agnostic inference
- Performance optimization
- Edge device compatibility

## 📁 Project Structure
```
linear_regression/
├── python/
│   ├── export_multi_backend.py  # Multi-backend model export
│   ├── benchmark.py             # Performance comparison
│   └── requirements.txt
├── cpp/
│   ├── main_multi_backend.cpp   # Multi-backend C++ demo
│   ├── CMakeLists_multi.txt     # Build configuration
│   └── build_multi.sh           # Build script
├── models/                      # Generated .pte files
└── README.md                    # This file
```