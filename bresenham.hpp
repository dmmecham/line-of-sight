#ifndef BRESENHAM_HPP
#define BRESENHAM_HPP

#include <cstdint>

#include "cuda_runtime.h"

extern __host__ __device__ bool isVisible(int16_t x1, int16_t y1, int16_t x2, int16_t y2, const int16_t* heightMap, int16_t width);

#endif // BRESENHAM_HPP