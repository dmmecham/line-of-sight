#ifndef GPU_HPP
#define GPU_HPP

#include <cstdint>
#include <vector>

std::vector<int32_t>* gpu(int16_t* input, size_t height, size_t width, size_t radius);

#endif // GPU_HPP
