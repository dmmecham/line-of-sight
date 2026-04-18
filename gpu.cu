#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include <stdio.h>

#include "cuda_utilities.hpp"
#include "file_utilities.hpp"
#include "line_of_sight_kernel.cu"


void gpu(std::string inputFilePath, std::string outputFilePath, size_t height, size_t width, size_t radius) {
  std::vector<int16_t> input = readFile(inputFilePath, height, width);
  
  CudaEngine<int16_t, int32_t> engine(lineOfSightKernel, height, width, radius, 0, height);
  
  int32_t* data = engine.compute(input.data());
  std::vector<int32_t> output(data, data + height * width * sizeof(int32_t));
  std::cout << output.size() << std::endl;
  
  std::cout << "GPU Time: " << std::fixed << std::setprecision(2) << engine.getTime() << " ms" << std::endl;

  writeFile(outputFilePath, output);

  delete data;
}
