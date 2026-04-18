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
#include "file_utilities.hpp"

#include "gpu.hpp"

__global__ void lineOfSightKernel(int16_t* input, int32_t* output, size_t height, size_t width, size_t radius) {
  size_t x1 = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y1 = blockIdx.y * blockDim.y + threadIdx.y;

  if (x1 >= width || y1 >= height) {
    return;
  }

  int32_t xStart = std::max((int32_t)x1 - (int32_t)radius, 0);
  int32_t xEnd = std::min((int32_t)x1 + (int32_t)radius , (int32_t)width - 1);

  int32_t yStart = std::max((int32_t)y1 - (int32_t)radius, 0);
  int32_t yEnd = std::min((int32_t)y1 + (int32_t)radius, (int32_t)height - 1);

  int32_t visiblePoints = 0;
  
  for (size_t y2 = yStart; y2 <= yEnd; y2++) {
    for (size_t x2 = xStart; x2 <= xEnd; x2++) {
      if (!(y2 == y1 && x2 == x1)) {
        visiblePoints += isVisible(x1, y1, x2, y2, input, width);
      }
    }
  }
  output[y1 * width + x1] = visiblePoints;
  if (y1 % 64 == 0 && x1 == 0) {
    printf("Processing row %llu\n", y1);
  }
}

void gpu(std::string inputFilePath, std::string outputFilePath, size_t height, size_t width, size_t radius) {
  std::vector<int16_t> input = readFile(inputFilePath, height, width);
  
  CudaEngine<int16_t, int32_t> engine(lineOfSightKernel, height, width, radius);
  
  int32_t* data = engine.compute(input.data());
  std::vector<int32_t> output(data, data + (height * width));
  std::cout << output.size() << std::endl;
  
  std::cout << "GPU Time: " << std::fixed << std::setprecision(2) << engine.getTime() << " ms" << std::endl;

  writeFile(outputFilePath, output);

  delete data;
}

#endif // GPU_CU
