cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GEMV LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set C++ standard (optional but recommended)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure we target at least one CUDA architecture to satisfy CMP0104.
# RTX 5090 reports compute capability 12.0, which maps to SM 120.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# Define the executable from your .cu file
add_executable(run main.cu)

# This makes sure -fPIC and similar flags are passed properly,
# which is needed by some Torch builds
set_property(TARGET run PROPERTY POSITION_INDEPENDENT_CODE ON)
