#ifndef GPU_CU
#define GPU_CU

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include <stdio.h>

#include "bresenham.hpp"
#include "cuda_utilities.hpp"

#include "gpu.hpp"

__global__ void lineOfSightKernel(int16_t* input, int32_t* output, size_t height, size_t width, size_t radius) {
  size_t x1 = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y1 = blockIdx.y * blockDim.y + threadIdx.y;

  if (x1 >= width || y1 >= height) {
    return;
  }

  size_t xStart = std::max(x1 - radius, (size_t)0);
  size_t xEnd = std::min(x1 + radius , width - 1);

  size_t yStart = std::max(y1 - radius, (size_t)0);
  size_t yEnd = std::min(y1 + radius, height - 1);

  int32_t visiblePoints = 0;
  
  for (size_t y2 = yStart; y2 <= yEnd; y2++) {
    for (size_t x2 = xStart; x2 <= xEnd; x2++) {
      if (!(y2 == y1 && x2 == x1)) {
        visiblePoints += isVisible(x1, y1, x2, y2, input, width);
      }
    }
  }
  output[y1 * width + x1] = visiblePoints;

  if (x1 == 0 && y1 == 0) {
    printf("x1: %llu y1: %llu xStart: %llu xEnd: %llu yStart: %llu yEnd: %llu visible: %d\n", x1, y1, xStart, xEnd, yStart, yEnd, visiblePoints);
  }
}

std::vector<int32_t>* gpu(int16_t* input, size_t height, size_t width, size_t radius) {
  CudaEngine<int16_t, int32_t> engine(lineOfSightKernel, height, width, radius);
  
  std::vector<int32_t>* output = new std::vector<int32_t>(*engine.compute(input));
  
  std::cout << "GPU Time: " << std::fixed << std::setprecision(2) << engine.getTime() << " ms" << std::endl;

  return output;
}

#endif // GPU_CU
