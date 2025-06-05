cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(executorch_mv2_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

add_subdirectory("../../executorch" executorch)

set(EXECUTORCH_BUILD_BUCK2 OFF CACHE BOOL "Disable Buck2 build" FORCE)
set(BUCK OFF CACHE BOOL "Disable Buck2" FORCE)

set(EXECUTORCH_USE_CMAKE_PACKAGE_BUILD ON CACHE BOOL "Use CMake package build" FORCE)

set(EXECUTORCH_BUILD_KERNELS_OPTIMIZED ON CACHE BOOL "Enabled Kernels Optimized" FORCE)
set(EXECUTORCH_BUILD_XNNPACK ON CACHE BOOL "Disable XNNPACK" FORCE)

set(EXECUTORCH_ENABLE_LOGGING ON CACHE BOOL "Enable logging" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "Enable data loader" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "Enable module extension" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "Enable tensor extension" FORCE)

set(DEMO_SOURCES main.cpp)

add_executable(executorch_mv2_demo_app ${DEMO_SOURCES})

target_include_directories(executorch_mv2_demo_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
  executorch_mv2_demo_app
  PRIVATE executorch
  extension_module_static
  extension_tensor
  optimized_native_cpu_ops_lib
  xnnpack_backend
)

set_target_properties(executorch_mv2_demo_app
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
