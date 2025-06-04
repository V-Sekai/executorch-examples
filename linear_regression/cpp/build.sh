#!/bin/bash
set -e

# Navigate to the repo root
SCRIPT_DIR="$(dirname "$0")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../" && pwd)"
cd "$REPO_ROOT"

# Create build directory at repo root
mkdir -p build
cd build

# Configure CMake pointing to the cpp source
cmake -DCMAKE_BUILD_TYPE=Release ../linear_regression/cpp

# Build the project
cmake --build . -j$(nproc)

echo "Build complete! Executable located at: ./linear_regression_example"
