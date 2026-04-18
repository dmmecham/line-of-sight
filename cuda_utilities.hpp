#ifndef CUDA_UTILITIES_HPP
#define CUDA_UTILITIES_HPP

#include <cmath>

#include "cuda_runtime.h"

// const int BLOCK_SIZE = 16; // The size of the block for the GPU kernel.
// const int MASK_SIZE = 100;
// const int MASK_RADIUS = MASK_SIZE / 2; // The radius of the neighborhood, which is used to determine how much extra space is needed in the tile for the shared memory kernel.
// const int TILE_SIZE = BLOCK_SIZE + MASK_SIZE - 1; // The size of the tile that each block will process.

// CUDA Error Checking Macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
    if (abort) exit(code);
  }
}

// Helper class for tracking time of GPU computation.
class CudaTimer {
public:
  CudaTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~CudaTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void startTimer() {
    cudaEventRecord(start);
  }

  void stopTimer() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
  }

  float getElapsedTime() {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
  }

  cudaEvent_t start;
  cudaEvent_t stop;
};

template <typename INPUT_DATA_TYPE, typename OUTPUT_DATA_TYPE>
class CudaEngine {
public:
  CudaEngine(void (*kernel)(INPUT_DATA_TYPE*, OUTPUT_DATA_TYPE*, size_t, size_t, size_t, size_t, size_t), size_t height, size_t width, size_t radius, size_t rowStart = 0, size_t rowEnd = 0)
  : height(height),
    kernel(kernel),
    radius(radius),
    rowEnd(rowEnd),
    rowStart(rowStart),
    width(width)
  {
    if (rowEnd == 0) {
      rowEnd = height;
    }
    std::cout << "height: " << height << " width: " << width << " radius " << radius
              << " rowStart: " << rowStart << " rowEnd: " << rowEnd << std::endl;
  }

  // Runs the kernel on the input data.
  OUTPUT_DATA_TYPE* compute(INPUT_DATA_TYPE* input_h) {
    // Start measuring how long the computations take, including setup and teardown.
    timer.startTimer();

    size_t numberOfPixels = height * width;
    size_t numberOfPixelsInRow = (rowEnd - rowStart) * width;
    // Pointers for synchronizing device memory.
    size_t inputSize = numberOfPixels * sizeof(INPUT_DATA_TYPE);
    size_t outputSize = numberOfPixelsInRow * sizeof(OUTPUT_DATA_TYPE);

    std::cout << "GPU compute output size: " << outputSize << std::endl;
    
    // Allocate device memory for the current input and the temporary input.
    INPUT_DATA_TYPE* input_d;
    OUTPUT_DATA_TYPE* output_d;
    gpuErrchk(cudaMalloc(&input_d, inputSize));
    gpuErrchk(cudaMalloc(&output_d, outputSize));
    // Copy the initial input from host to device memory.
    gpuErrchk(cudaMemcpy(input_d, input_h, inputSize, cudaMemcpyHostToDevice));
    // Make sure data fully copied to the device before launching the kernel.
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "Error: Data copy to device failed." << std::endl;
      return nullptr;
    }
    gpuErrchk(cudaPeekAtLastError());

    // Determine the optimal block and grid size by querying the GPU.
    // int blockSize = 0;
    // int minimumGridSize = 0;
    // cudaOccupancyMaxPotentialBlockSize(&minimumGridSize, &blockSize, kernel, 0, 0);
    
    // Launch the kernel to compute the output.
    std::cout << height << "x" << width << " with radius " << radius << " start: " << rowStart << " rowEnd: " << rowEnd << std::endl;
  
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
      (height + blockDim.y - 1) / blockDim.y);

    kernel<<<gridDim, blockDim>>>(input_d, output_d, height, width, radius, rowStart, rowEnd);
    gpuErrchk(cudaPeekAtLastError());
    // Copy the output back from the device to the host.
    OUTPUT_DATA_TYPE* output_h = new OUTPUT_DATA_TYPE[outputSize];
    gpuErrchk(cudaMemcpy(output_h, output_d, outputSize, cudaMemcpyDeviceToHost));
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "Error: Data copy to device failed." << std::endl;
      return nullptr;
    }
    // Clean up device memory.
    cudaFree(input_d);
    cudaFree(output_d);
    timer.stopTimer();

    return output_h;
  }

  // Determine how long the computation took.
  float getTime() {
    return timer.getElapsedTime();
  }

private:
  // Dimensions.
  size_t height;
  size_t radius;
  size_t width;
  // Offsets for rows to process.
  size_t rowEnd;
  size_t rowStart;
  // Timing metrics.
  CudaTimer timer;
  // Reference to the kernel function to be executed.
  void (*kernel)(INPUT_DATA_TYPE*, OUTPUT_DATA_TYPE*, size_t, size_t, size_t, size_t, size_t);
};

#endif // CUDA_UTILITIES_HPP
