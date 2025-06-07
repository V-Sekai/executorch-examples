# ExecuTorch Linear Regression Demo C++ Application

This is a simple C++ demo application that uses the ExecuTorch library for linear regression model inference.

## Build instructions

0. Export the model. See [linear_regression/python/README.md](../python/README.md)

1. Build the project from the repository root:
   ```bash
   cd ~/executorch-examples
   ./linear_regression/cpp/build.sh
   ```

2. Run the demo application:
   ```bash
   ./build/bin/linear_regression_example
   ```

## Dependencies

- CMake 3.18 or higher
- C++17 compatible compiler

## Notes

- Make sure you have the correct model file (`.pte`) compatible with ExecuTorch.
- The model file should be located at `linear_regression/models/simple_linear.pte`.
- This demo uses predefined input features [1, 2, 3, 4]. In a real application, you would replace this with actual input data.
