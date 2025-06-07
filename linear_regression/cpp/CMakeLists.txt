cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(executorch_multi_backend_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# Disable Buck2 build system
set(EXECUTORCH_BUILD_BUCK2 OFF CACHE BOOL "Disable Buck2 build" FORCE)
set(BUCK OFF CACHE BOOL "Disable Buck2" FORCE)

# Use CMake package build
set(EXECUTORCH_USE_CMAKE_PACKAGE_BUILD ON CACHE BOOL "Use CMake package build" FORCE)

# Enable all backends
set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "Enable Kernels Optimized" FORCE)
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "Enable XNNPACK" FORCE)

# Enable Vulkan if available
find_package(Vulkan QUIET)
if(Vulkan_FOUND)
    set(EXECUTORCH_BUILD_VULKAN ON CACHE BOOL "Enable Vulkan" FORCE)
    message(STATUS "Vulkan found, enabling Vulkan backend")
else()
    set(EXECUTORCH_BUILD_VULKAN OFF CACHE BOOL "Disable Vulkan" FORCE)
    message(STATUS "Vulkan not found, disabling Vulkan backend")
endif()

if(APPLE)
    set(EXECUTORCH_BUILD_MPS ON CACHE BOOL "Enable MPS" FORCE)
    message(STATUS "macOS detected, enabling MPS backend")
else()
    set(EXECUTORCH_BUILD_MPS OFF CACHE BOOL "Disable MPS" FORCE)
    message(STATUS "Not macOS, disabling MPS backend")
endif()

FetchContent_Declare(
    executorch
    GIT_REPOSITORY https://github.com/pytorch/executorch.git
    GIT_TAG v0.6.0
    SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/executorch
)
FetchContent_MakeAvailable(executorch)

# Create executable
add_executable(executorch_multi_backend_demo main_multi_backend.cpp)

# Link libraries
target_link_libraries(executorch_multi_backend_demo
    executorch
    extension_module
    extension_tensor
    optimized_native_cpu_ops_lib
)

# Link XNNPACK backend
if(TARGET xnnpack_backend)
    target_link_libraries(executorch_multi_backend_demo xnnpack_backend)
endif()

# Link Vulkan backend if available
if(TARGET vulkan_backend)
    target_link_libraries(executorch_multi_backend_demo vulkan_backend)
endif()

# Link MPS backend if available (macOS only)
if(TARGET mps_backend)
    target_link_libraries(executorch_multi_backend_demo mps_backend)
endif()

# Set output directory
set_target_properties(executorch_multi_backend_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)