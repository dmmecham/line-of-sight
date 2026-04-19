# line-of-sight
[https://github.com/dmmecham/line-of-sight](GitHub Repository)

Calculates the number of pixels visible for every pixel in an elevation map within a provided radius. Utilizes Bresenham's algorithm over a variety of computation types that split the task up for different performance gains.

## Build
### Prerequisites
- CMake 3.21+
- C++ compiler (C++17 support, Visual Studio 2022/2026 and gcc 8 have been tested)
- MPI
- NVIDIA CUDA SDK

For the CHPC lab, the following command will load all necessary modules: `module load cmake cuda gcc mpi`

This project utilizes CMake to generate a build system suitable for the platform and compiler of choice. If utilizing the CLI, and assuming that the source repositorya and the build directory are siblings, the following command run from the build directory will create the appropriate build target: `cmake ../source`. Then simply run `make` (or the analogous command/solution file for the platform/compiler) to build the executables. If only `line_of_sight` is desired, `make line_of_sight` will only build that executable.

## Run
The program requires the following arguments (this is displayed if not all are provided):
- `input elevation file path` (2D elevation stored as int16_t for each point)
- `output visibility file path` (2D visibility stored as int32_t for each visible point in a surrounding radius)
- `height` of 2D elevation data
- `width` of 2D elevation data
- `radius` (number of pixels to calculate around each point, higher values take more time to process)
- `compute type`: no-threads, threads, gpu, mpi, mpi-gpu

Example 1: `./line_of_sight ../source/srtm_14_04_6000x6000_short16.raw ./gpu_output.raw 6000 6000 100 gpu` would run the GPU algorithm over the input 6000x6000 image with a radius of 100 and save to the output path.

Example 2: Utilizing MPI requires a prefix: `mpiexec -n 4 ./line_of_sight ../source/srtm_14_04_6000x6000_short16.raw ./mpi_output.raw 6000 6000 100 mpi` would run the MPI algorithm over the input 6000x6000 image with a radius of 100 and save to the output path.

If all provided arguments are correct, the program will write the output file at completion.

The `compute type` argument determines how processing is parallelized:
- `no-threads` is the basic CPU serialized calculation and is the slowest.
- `threads` utilizes OpenMP to spread the calculations over CPU threads/cores.
- `gpu` utilizes NVIDIA CUDA to calculate on a compatible GPU (this is the fastest available computation on a single system).
- `mpi` spreads the CPU serialized over several processing units. In order to take advantage of MPI, preface the command with `mpiexec -n #`, where `#` is the number of processing units. This should be faster than `no-threads` on a single system and comparable to `threads` on the same system, but should allow more resources that the source system's CPU can provide.
- `mpi-gpu` spreads the task over multiple processing units via MPI and uses the NVIDIA CUDA algorithm to do processing on each GPU. Ideally this can provide the same increase in speed as MPI with multiplied speed of `gpu`, but the overhead of using both together does decrease the improvement. This will not see any improvement over `gpu` if all processing units are on the same system.

## Validation
All compute types should produce the same output data if all other arguments (besides the output file path) are the same. A validation program is provided that simply compares two files and checks if they are identical in data (no file metadata is checked with the file system). `verification <file path 1> <file path 2>` will indicate whether the files are identical. It will either be built when executing `make` with no arguments, or `make verification` to specifically build this executable.

## Contributors
- Matt Robinson provided the initial Bresenham modularized implementation, the serialized and threaded implementations, much bug fixing on the other implementations, testing and validation of output for each implementation, and performance metrics.
- Drew Mecham provided the CMake build implementation, documentation, initial GPU kernel, MPI  initial implementation, the combined MPI/GPU initial implementation, verification program, and testing.
