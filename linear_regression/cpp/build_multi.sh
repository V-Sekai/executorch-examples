#!/bin/bash
set -e

echo "🏗️  Building ExecuTorch Multi-Backend Linear Regression Demo"

# Navigate to the repo root
SCRIPT_DIR="$(dirname "$0")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$REPO_ROOT"

# Create build directory at repo root
mkdir -p build_multi
cd build_multi

# Configure CMake with multi-backend support
echo "⚙️  Configuring CMake with multi-backend support..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=17 \
      ../linear_regression/cpp

# Build the project
echo "🔨 Building project..."
cmake --build . --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo "✅ Build complete!"
echo "📁 Executable location: ./bin/executorch_multi_backend_demo"

# Check if models exist
echo ""
echo "📋 Checking for model files..."
MODELS_DIR="$REPO_ROOT/linear_regression/models"

if [ -d "$MODELS_DIR" ]; then
    echo "📂 Models directory found:"
    ls -la "$MODELS_DIR"/*.pte 2>/dev/null || echo "   ⚠️  No .pte files found"
else
    echo "❌ Models directory not found. Run Python export first:"
    echo "   cd linear_regression/python && python export_multi_backend.py"
fi

echo ""
echo "🚀 To run the demo:"
echo "   ./bin/executorch_multi_backend_demo"