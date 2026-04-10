cmake_minimum_required(VERSION 3.18)
project(LineOfSight)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
# Required external packages.
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)

# Handle CUDA compilation and linking.
enable_language(CUDA)
# Disable until the file(s) are added.
# add_library(Kernels
#   kernel.cu
# )
# set_target_properties(Kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(Kernels PUBLIC CUDA::cudart)

# Main executable with linkage to Kernels.
add_executable(LineOfSight
  bresemham.hpp
  line_of_sight.cpp
)
target_link_libraries(LineOfSight PRIVATE
#  Kernels
  CUDA::cudart
  MPI::MPI_CXX
)
