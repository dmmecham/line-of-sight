#ifndef LINE_OF_SIGHT_CU
#define LINE_OF_SIGHT_CU

#include <cmath>
#include <iostream>

#include "bresenham.hpp"

__global__ inline void lineOfSightKernel(int16_t* input, int32_t* output, size_t height, size_t width, size_t radius, size_t rowStart, size_t rowEnd) {
  size_t x1 = blockIdx.x * blockDim.x + threadIdx.x;
  size_t y1 = blockIdx.y * blockDim.y + threadIdx.y;

  if (x1 >= width || y1 >= rowEnd || y1 < rowStart) {
    return;
  }

  int32_t xStart = std::max((int32_t)x1 - (int32_t)radius, 0);
  int32_t xEnd = std::min((int32_t)x1 + (int32_t)radius , (int32_t)width - 1);

  int32_t yStart = std::max((int32_t)y1 - (int32_t)radius, (int32_t)rowStart);
  int32_t yEnd = std::min((int32_t)y1 + (int32_t)radius, (int32_t)rowEnd - 1);

  int32_t visiblePoints = 0;
  
  for (size_t y2 = yStart; y2 <= yEnd; y2++) {
    for (size_t x2 = xStart; x2 <= xEnd; x2++) {
      if (!(y2 == y1 && x2 == x1)) {
        visiblePoints += isVisible(x1, y1, x2, y2, input, width);
      }
    }
  }
  output[(y1 - yStart) * width + x1] = visiblePoints;
  if (y1 % 64 == 0 && x1 == 0) {
    printf("Processing row %llu\n", y1);
  }
}

#endif // LINE_OF_SIGHT_CU
