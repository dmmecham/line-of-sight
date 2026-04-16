#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

//#include <cuda_runtime.h>

#include "gpu.hpp"
//#include "mpi.hpp"
//#include "mpi_gpu.hpp"
#include "no_threads.hpp"
//#include "threads.hpp"


enum ComputeType {
  NO_THREADS,
  THREADS,
  GPU,
  MPI,
  MPI_GPU
};

std::map<std::string, ComputeType> computeTypeMap = {
  {"no-thread", NO_THREADS},
  //{"threads", THREADS},
  {"gpu", GPU},
  //{"mpi", MPI},
  //{"mpi-gpu", MPI_GPU}
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
    size_t width = std::atoll(argv[3]);
    size_t height = std::atoll(argv[4]);
    size_t radius = std::atoll(argv[5]);
    ComputeType computeType = computeTypeMap[argv[6]];

    size_t pixels = height * width;
    size_t dataSize = pixels * sizeof(int16_t);

    // Read input file once
    std::ifstream inputFile(inputFilePath, std::ios::binary | std::ios::ate);
    if (!inputFile.is_open()) return 1;

    std::streamsize totalBytes = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    if (totalBytes != dataSize) {
        std::cerr << "Error: Input file size does not match expected dimensions." << std::endl;
        std::cerr << "Expected " << dataSize << " bytes, got " << totalBytes << " bytes." << std::endl;
        return 1;
    }

    std::vector<int16_t> heightMap(pixels);
    if (!inputFile.read(reinterpret_cast<char*>(heightMap.data()), totalBytes)) {
        std::cerr << "Error reading input file." << std::endl;
        return 1;
    }

    std::cout << "Read height map from: " << inputFilePath << std::endl;
    std::cout << "Dimensions: " << width << "x" << height << ", Radius: " << radius << std::endl;
    std::cout << "Compute Type: " << argv[6] << std::endl;
    std::cout << totalBytes << " bytes read." << std::endl;
    std::cout << dataSize << " bytes expected." << std::endl;

    std::vector<int32_t>* output;

    switch (computeType) {
      case ComputeType::NO_THREADS:
        // Run the serial algorithm implementation
        output = serialAlgorithm(heightMap, height, width, radius);
        break;
      // case ComputeType::THREADS:
      //   output = threadedAlgoritm(heightMap, height, width, radius);
      //   break;
      case ComputeType::GPU:
        output = gpu(heightMap.data(), height, width, radius);
        break;
      // case ComputeType::MPI:
      //   output = mpiAlgorith(heightMap, height, width, radius);
      //   break;
      // case ComputeType::MPI_GPU:
      //   output = mpiGpuAlgorithm(heightMap, height, width, radius);
      //   break;
      default:
        std::cerr << "Invalid compute type" << std::endl;
        return 1;
    }

    // Write output file as 32-bit signed ints
    std::ofstream outFile(outputFilePath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << outputFilePath << std::endl;
        return 1;
    }

    outFile.write(reinterpret_cast<const char*>(output->data()), output->size());
    outFile.close();

    std::cout << "Wrote visibility counts to: " << outputFilePath << std::endl;

    return 0;
}
