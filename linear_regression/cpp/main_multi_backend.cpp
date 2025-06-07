#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>

using namespace ::executorch::extension;
namespace fs = std::filesystem;

struct BackendResult {
    std::string name;
    bool available;
    double inference_time_ms;
    float output_value;
};

class MultiBackendDemo {
private:
    std::vector<std::pair<std::string, std::string>> backends = {
        {"../../linear_regression/models/linear_mps.pte", "MPS (Apple Metal)"},
        {"../../linear_regression/models/linear_vulkan.pte", "Vulkan (GPU)"},
        {"../../linear_regression/models/linear_xnnpack.pte", "XNNPACK (CPU)"},
        {"../../linear_regression/models/linear_portable.pte", "Portable (Reference)"}
    };
    
    float input_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};

public:
    BackendResult testBackend(const std::string& model_path, const std::string& name) {
        BackendResult result = {name, false, 0.0, 0.0f};
        
        // Check if model file exists
        if (!fs::exists(model_path)) {
            std::cout << "❌ " << name << ": Model file not found (" << model_path << ")" << std::endl;
            return result;
        }
        
        try {
            // Load model
            Module module(model_path);
            auto tensor = from_blob(input_data, {1, 4});
            
            // Warmup
            for (int i = 0; i < 10; i++) {
                auto warmup_result = module.forward(tensor);
                (void)warmup_result; // Suppress unused variable warning
            }
            
            // Benchmark
            auto start = std::chrono::high_resolution_clock::now();
            const int iterations = 1000;
            
            auto forward_result = module.forward(tensor);
            if (!forward_result.ok()) {
                std::cout << "❌ " << name << ": Model inference failed" << std::endl;
                return result;
            }
            
            for (int i = 0; i < iterations - 1; i++) {
                auto benchmark_result = module.forward(tensor);
                (void)benchmark_result; // Suppress unused variable warning
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            result.available = true;
            result.inference_time_ms = duration.count() / 1000.0 / iterations;
            result.output_value = forward_result->at(0).toTensor().const_data_ptr<float>()[0];
            
            std::cout << "✅ " << name << ": " << result.inference_time_ms << " ms avg" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "❌ " << name << ": Exception - " << e.what() << std::endl;
        }
        
        return result;
    }
    
    void runDemo() {
        std::cout << "🚀 ExecuTorch Multi-Backend Linear Regression Demo" << std::endl;
        std::cout << "=" << std::string(55, '=') << std::endl;
        std::cout << "📊 Input features: [" << input_data[0] << ", " << input_data[1] 
                  << ", " << input_data[2] << ", " << input_data[3] << "]" << std::endl;
        std::cout << std::endl;
        
        std::vector<BackendResult> results;
        
        // Test all backends
        for (const auto& backend : backends) {
            auto result = testBackend(backend.first, backend.second);
            results.push_back(result);
        }
        
        // Find available backends
        std::vector<BackendResult> available_results;
        for (const auto& result : results) {
            if (result.available) {
                available_results.push_back(result);
            }
        }
        
        if (available_results.empty()) {
            std::cout << std::endl << "❌ No backends available! Check model files." << std::endl;
            return;
        }
        
        // Sort by performance (fastest first)
        std::sort(available_results.begin(), available_results.end(),
                  [](const BackendResult& a, const BackendResult& b) {
                      return a.inference_time_ms < b.inference_time_ms;
                  });
        
        // Display results
        std::cout << std::endl << "🏆 Performance Ranking:" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (size_t i = 0; i < available_results.size(); i++) {
            const auto& result = available_results[i];
            std::string rank = (i == 0) ? "🥇" : (i == 1) ? "🥈" : (i == 2) ? "🥉" : "📊";
            
            printf("%s %-25s %8.3f ms (output: %.6f)\n", 
                   rank.c_str(), result.name.c_str(), 
                   result.inference_time_ms, result.output_value);
        }
        
        // Best backend summary
        const auto& fastest = available_results[0];
        const auto& slowest = available_results.back();
        
        std::cout << std::endl << "📈 Summary:" << std::endl;
        std::cout << "   🏃 Fastest: " << fastest.name << std::endl;
        std::cout << "   ⚡ Speed improvement: " << (slowest.inference_time_ms / fastest.inference_time_ms) 
                  << "x over slowest" << std::endl;
        std::cout << std::endl << "✅ ExecuTorch multi-backend demo completed successfully!" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    MultiBackendDemo demo;
    demo.runDemo();
    return 0;
}