cmake_minimum_required(VERSION 3.21)
project(LineOfSight LANGUAGES C CXX CUDA)
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Required external packages.
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(line_of_sight
  bresenham.cu
  bresenham.hpp
  cuda_utilities.hpp
  gpu.cu
  gpu.hpp
  line_of_sight_kernel.cu
  main.cpp
  mpi_gpu.cu
  mpi_gpu.hpp
  no_threads.hpp
  threads.hpp
  mpi.hpp
)
set_target_properties(line_of_sight PROPERTIES
  CUDA_ARCHITECTURES native    
  CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(line_of_sight PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_link_libraries(line_of_sight PRIVATE
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  CUDA::cudart
)

add_executable(verification
  verification.cpp
)
