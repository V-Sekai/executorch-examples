#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <iostream>

using namespace ::executorch::extension;

int main(int argc, char* argv[]) {
    Module module("linear_regression/models/simple_linear.pte");

    float input[1 * 4] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto tensor = from_blob(input, {1, 4});

    const auto result = module.forward(tensor);

    if (result.ok()) {
        const auto output = result->at(0).toTensor().const_data_ptr<float>();
        std::cout << "🎉 SUCCESS: Linear regression inference completed!" << std::endl;
        std::cout << "📊 Input features: [" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << "]" << std::endl;
        std::cout << "🎯 Predicted output: " << output[0] << std::endl;
        std::cout << "✅ ExecuTorch linear regression model is working correctly!" << std::endl;
    } else {
        std::cout << "❌ ERROR: Model inference failed!" << std::endl;
        return 1;
    }
    
    return 0;
}
