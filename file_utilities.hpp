#ifndef FILE_UTILITIES_HPP
#define FILE_UTILITIES_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

inline std::vector<int16_t>* readFile(std::string filePath, size_t height, size_t width) {
  size_t pixels = height * width;
  size_t dataSize = pixels * sizeof(int16_t);

  // Read input file once
  std::ifstream inputFile(filePath, std::ios::binary | std::ios::ate);
  if (!inputFile.is_open()) throw;

  std::streamsize totalBytes = inputFile.tellg();
  inputFile.seekg(0, std::ios::beg);
  
  if (totalBytes != dataSize) {
      std::cerr << "Error: Input file size does not match expected dimensions." << std::endl;
      std::cerr << "Expected " << dataSize << " bytes, got " << totalBytes << " bytes." << std::endl;
      throw;
  }

  std::cout << totalBytes << " bytes read." << std::endl;
  std::cout << dataSize << " bytes expected." << std::endl;

  std::vector<int16_t>* heightMap = new std::vector<int16_t>(pixels);
  if (!inputFile.read(reinterpret_cast<char*>(heightMap->data()), totalBytes)) {
      std::cerr << "Error reading input file." << std::endl;
      throw;
  }

  
  return heightMap;
}

inline void writeFile(std::string filePath, std::vector<int32_t>* data) {
  // Write output file as 32-bit signed ints
    std::ofstream outFile(filePath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << filePath << std::endl;
        throw;
    }

    if (!outFile.write(reinterpret_cast<const char*>(data->data()), data->size())) {
      std::cerr << "Failed to write           output file: " << filePath << std::endl;
      throw;
    }
    outFile.close();

    std::cout << "Wrote visibility counts to: " << filePath << std::endl;
}

#endif // FILE_UTILITIES_HPP
