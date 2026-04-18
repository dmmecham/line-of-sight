#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

//#include <cuda_runtime.h>

#include "gpu.hpp"
#include "mpi.hpp"
#include "mpi_gpu.hpp"
#include "no_threads.hpp"
#include "threads.hpp"


enum ComputeType {
  NO_THREADS,
  THREADS,
  GPU,
  MPI,
  MPI_GPU
};

std::map<std::string, ComputeType> computeTypeMap = {
  {"no-threads", NO_THREADS},
  {"threads", THREADS},
  {"gpu", GPU},
  {"mpi", MPI},
  {"mpi-gpu", MPI_GPU}
};

void printUsage(char* processName) {
  std::string computeTypes;
  

  std::cerr << "Usage: "
            << processName << " "
            << "<input_file.raw> "
            << "<output_file.raw> "
            << "<height> "
            << "<width> "
            << "<radius> "
            << "<computeType: [";

  // Extract string values from the map.
  bool first = true;
  for (const auto& [key, value] : computeTypeMap) {
    if (!first) {
      std::cerr << ", ";
    }
    std::cerr << key;
  }

  std::cerr << "]>" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        printUsage(argv[0]);
        return 1;
    }

    std::string inputFilePath = argv[1];
    std::string outputFilePath = argv[2];
    size_t height = std::atoll(argv[3]);
    size_t width = std::atoll(argv[4]);
    size_t radius = std::atoll(argv[5]);
    ComputeType computeType = computeTypeMap[argv[6]];

    std::cout << "Read height map from: " << inputFilePath << std::endl;
    std::cout << "Dimensions: " << width << "x" << height << ", Radius: " << radius << std::endl;
    std::cout << "Compute Type: " << argv[6] << std::endl;

    try {
      switch (computeType) {
        case ComputeType::NO_THREADS:
          // Run the serial algorithm implementation
          serialAlgorithm(inputFilePath, outputFilePath, height, width, radius);
          break;
        case ComputeType::THREADS:
          threadedAlgorithm(inputFilePath, outputFilePath, height, width, radius);
          break;
        case ComputeType::GPU:
          gpu(inputFilePath, outputFilePath, height, width, radius);
          break;
        case ComputeType::MPI:
          mpiAlgorithm(inputFilePath, outputFilePath, height, width, radius);
          break;
        case ComputeType::MPI_GPU:
          mpiGpuAlgorithm(inputFilePath, outputFilePath, height, width, radius);
          break;
        default:
          std::cerr << "Invalid compute type" << std::endl;
          return 1;
      }
    } catch (const std::exception& e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }

    return 0;
}
