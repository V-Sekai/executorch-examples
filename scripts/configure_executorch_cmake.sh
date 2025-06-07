#!/bin/bash
set -ex
python_executable=$(which python)
site_packages_dir=$(python -c "import site; import os; print(os.path.realpath(site.getsitepackages()[0]))")
echo "Using Python executable: $python_executable"
echo "Installing to site-packages: $site_packages_dir"
mkdir -p cmake-build
cd cmake-build
# Remove CMakeCache.txt and CMakeFiles/ to ensure a clean configure
rm -rf CMakeCache.txt CMakeFiles/
cmake .. \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$site_packages_dir" \
    -DEXECUTORCH_BUILD_PYBIND=ON \
    -DEXECUTORCH_BUILD_SDK=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TRAINING=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DPYTHON_EXECUTABLE="$python_executable"
