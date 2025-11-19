cmake_minimum_required(VERSION 3.10)

project(auto_cuda_debugger_kernels LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)

# Specify the CUDA files to be compiled
set(KERNELS
    mha.cu
    gatemlp.cu
)

# Create a library from the CUDA files
add_library(cuda_kernels SHARED ${KERNELS})

# Specify include directories
target_include_directories(cuda_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Link against necessary libraries (if any)
# target_link_libraries(cuda_kernels <libraries>)