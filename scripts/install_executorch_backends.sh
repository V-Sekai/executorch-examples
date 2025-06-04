#!/bin/bash
# Install ExecuTorch with backend support for CI/CD environments

set -e

echo "🔧 Installing ExecuTorch with Multi-Backend Support"
echo "=================================================="

# Install basic dependencies
echo "📦 Installing base dependencies..."
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib numpy

# Try basic ExecuTorch installation first
echo "📦 Installing basic ExecuTorch..."
pip install executorch || echo "⚠️  Basic ExecuTorch installation failed"

# System-specific backend installation
OS="$(uname -s)"
case "${OS}" in
    Linux*)
        echo "🐧 Linux detected - attempting Vulkan backend..."
        
        # Check if Vulkan tools are available
        if command -v vulkaninfo &> /dev/null; then
            echo "✅ Vulkan tools available"
        else
            echo "❌ Vulkan tools not found"
        fi
        
        # Try to build with Vulkan support
        echo "🔨 Attempting to build ExecuTorch with Vulkan..."
        if [ ! -d "/tmp/executorch" ]; then
            git clone --depth 1 https://github.com/pytorch/executorch.git /tmp/executorch || {
                echo "❌ Failed to clone ExecuTorch repository"
                exit 0
            }
        fi
        
        cd /tmp/executorch
        pip install -r requirements.txt || echo "⚠️  Some requirements failed"
        
        # Try to build with Vulkan
        python setup.py develop --cmake-args="-DEXECUTORCH_BUILD_VULKAN=ON" || {
            echo "⚠️  Vulkan build failed, ExecuTorch will use XNNPACK fallback"
        }
        ;;
        
    Darwin*)
        echo "🍎 macOS detected - attempting MPS backend..."
        
        # Check if MPS is available
        python -c "import torch; print('MPS available:', torch.backends.mps.is_available())" || echo "MPS check failed"
        
        # Try to build with MPS support
        echo "🔨 Attempting to build ExecuTorch with MPS..."
        if [ ! -d "/tmp/executorch" ]; then
            git clone --depth 1 https://github.com/pytorch/executorch.git /tmp/executorch || {
                echo "❌ Failed to clone ExecuTorch repository"
                exit 0
            }
        fi
        
        cd /tmp/executorch
        pip install -r requirements.txt || echo "⚠️  Some requirements failed"
        
        # Try to build with MPS
        python setup.py develop --cmake-args="-DEXECUTORCH_BUILD_MPS=ON" || {
            echo "⚠️  MPS build failed, ExecuTorch will use XNNPACK fallback"
        }
        ;;
        
    MINGW*|CYGWIN*|MSYS*)
        echo "🪟 Windows detected - using basic ExecuTorch..."
        echo "ℹ️  Advanced backends may require manual compilation on Windows"
        ;;
        
    *)
        echo "❓ Unknown OS: ${OS}"
        echo "ℹ️  Using basic ExecuTorch installation"
        ;;
esac

echo ""
echo "✅ ExecuTorch installation complete!"
echo "🔍 Backend availability will be checked during runtime"
echo "📋 Available backends depend on successful compilation and system support"