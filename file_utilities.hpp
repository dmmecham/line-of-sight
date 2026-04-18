#ifndef FILE_UTILITIES_HPP
#define FILE_UTILITIES_HPP

#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

inline std::vector<int16_t> readFile(std::string filePath, size_t height, size_t width) {
  size_t pixels = height * width;
  size_t dataSize = pixels * sizeof(int16_t);

  // Read input file once
  std::ifstream inputFile(filePath, std::ios::binary | std::ios::ate);
  if (!inputFile.is_open()) {
    throw std::runtime_error("Unable to open input file " + filePath);
  }

  std::streamsize totalBytes = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);
  
  if (totalBytes != dataSize) {
      std::stringstream error;
      error << "Error: Input file size does not match expected dimensions." << std::endl;
      error << "Expected " << dataSize << " bytes, got " << totalBytes << " bytes." << std::endl;
      throw std::runtime_error(error.str());
  }

  std::cout << totalBytes << " bytes read." << std::endl;
  std::cout << dataSize << " bytes expected." << std::endl;

  std::vector<int16_t> heightMap(pixels);
  if (!inputFile.read(reinterpret_cast<char*>(heightMap.data()), totalBytes)) {
    throw std::runtime_error("Error reading input file.");
  }
  
  return heightMap;
}

inline void writeFile(std::string filePath, const std::vector<int32_t>& data) {
  // Write output file as 32-bit signed ints
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile.is_open()) {
        std::stringstream error;
        error << "Failed to open output file: " << filePath << std::endl;
        throw std::runtime_error(error.str());
    }

    if (!outFile.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(int32_t))) {
      std::stringstream error;
      error << "Failed to write output file: " << filePath << std::endl;
      throw std::runtime_error(error.str());
    }
    outFile.close();

    std::cout << "Wrote visibility counts to: " << filePath << std::endl;
}

#endif // FILE_UTILITIES_HPP
