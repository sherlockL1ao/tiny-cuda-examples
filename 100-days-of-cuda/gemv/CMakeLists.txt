cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GEMV LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Try to locate the Python interpreter early so we can ask it where Torch's CMake
# package files live (torch.utils.cmake_prefix_path).
find_package(Python3 COMPONENTS Interpreter QUIET)

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

# Optional: build a Python-loadable Torch extension to call the kernel from PyTorch.
# Try to seed CMake's search path with torch.utils.cmake_prefix_path so users
# don't have to set Torch_DIR manually.
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch, sys; sys.stdout.write(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(TORCH_CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    message(STATUS "Added Torch CMake prefix path from Python: ${TORCH_CMAKE_PREFIX_PATH}")
  endif()
endif()

find_package(Torch QUIET)
if(Torch_FOUND)
  message(STATUS "Found Torch, building gemv_ext for Python")

  # Python headers are not pulled in automatically by Torch's CMake config.
  # Explicitly locate the interpreter + development headers so <Python.h> resolves.
  find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  # torch_python is needed for Python-facing bindings.
  find_library(TORCH_PYTHON_LIBRARY torch_python
    PATHS "${Python3_SITEARCH}/torch/lib" "${Python_SITEARCH}/torch/lib"
    NO_DEFAULT_PATH)

  add_library(gemv_ext SHARED gemv_binding.cu)

  target_link_libraries(gemv_ext PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY} Python3::Python)
  target_include_directories(gemv_ext PRIVATE ${TORCH_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS})

  # Ensures correct ABI flags and PIC for Python import.
  set_property(TARGET gemv_ext PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET gemv_ext PROPERTY CXX_STANDARD 17)
  set_property(TARGET gemv_ext PROPERTY CUDA_STANDARD 17)
  set_property(TARGET gemv_ext PROPERTY CUDA_SEPARABLE_COMPILATION ON)

  # Drop the .so next to main.py so `import gemv_ext` works without PYTHONPATH tweaks.
  set_target_properties(gemv_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  )

  # Drop the default "lib" prefix so Python can `import gemv_ext` directly.
  set_target_properties(gemv_ext PROPERTIES PREFIX "" SUFFIX ".so")
else()
  message(STATUS "Torch not found; skipping Python extension target")
endif()
