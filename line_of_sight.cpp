#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <chrono>
#include <cstdint>
#include <algorithm>
//#include <cuda_runtime.h>
#include "bresenham.hpp"

const int WIDTH = 6000; // Only dimensions of 6000 x 6000 pixels are supported.
const int GRID_SIZE = WIDTH * WIDTH;
const size_t BUFFER_SIZE = GRID_SIZE * sizeof(int16_t);
const int RADIUS = 10; // radius in pixels (results in (2*RADIUS+1)^2 window)

void serialAlgorithm(const std::vector<int16_t>& heightMap, std::vector<int32_t>& visibleCounts, int width, int radius);

// CUDA Error Checking Macro
/*#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}*/

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./line_of_sight <input_file.raw>" << std::endl;
        return 1;
    }

    std::string filePath = argv[1];

    // Read input file once
    std::ifstream inputFile(filePath, std::ios::binary | std::ios::ate);
    if (!inputFile.is_open()) return 1;

    std::streamsize totalBytes = inputFile.tellg();
    inputFile.seekg(0, std::ios::beg);
    if (totalBytes != BUFFER_SIZE) {
        std::cerr << "Error: Input file size does not match expected dimensions." << std::endl;
        std::cerr << "Expected " << BUFFER_SIZE << " bytes, got " << totalBytes << " bytes." << std::endl;
        return 1;
    }

    std::vector<int16_t> heightMap(GRID_SIZE);
    if (!inputFile.read(reinterpret_cast<char*>(heightMap.data()), totalBytes)) {
        std::cerr << "Error reading input file." << std::endl;
        return 1;
    }
    std::vector<int32_t> visibleCounts(GRID_SIZE, 0);

    // Run the serial algorithm implementation
    serialAlgorithm(heightMap, visibleCounts, WIDTH, RADIUS);

    // Write output file as 32-bit signed ints
    std::string outPath = filePath + ".visible.raw";
    std::ofstream outFile(outPath, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open output file: " << outPath << std::endl;
        return 1;
    }

    outFile.write(reinterpret_cast<const char*>(visibleCounts.data()), visibleCounts.size() * sizeof(int32_t));
    outFile.close();

    std::cout << "Wrote visibility counts to: " << outPath << std::endl;

    return 0;
}

// Serial implementation
void serialAlgorithm(const std::vector<int16_t>& heightMap, std::vector<int32_t>& visibleCounts, int width, int radius) {
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int y1 = 0; y1 < width; ++y1) {
        for (int x1 = 0; x1 < width; ++x1) {
            int idx1 = y1 * width + x1;
            int32_t count = 0;

            // Radius-limited window
            int y0 = std::max(0, y1 - radius);
            int yN = std::min(width - 1, y1 + radius);
            int x0 = std::max(0, x1 - radius);
            int xN = std::min(width - 1, x1 + radius);

            for (int y2 = y0; y2 <= yN; ++y2) {
                for (int x2 = x0; x2 <= xN; ++x2) {
                    // Skip self-comparison
                    if (x1 == x2 && y1 == y2) continue;
                    // Check line of sight using Bresenham's algorithm
                    if (isVisible(x1, y1, x2, y2, heightMap.data(), width)) ++count;
                }
            }

            visibleCounts[idx1] = count;
        }

        // Print progress every 64 rows
        if ((y1 % 64) == 0) {
            auto tnow = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(tnow - t0).count();
            std::cout << "Row " << y1 << " elapsed: " << sec << "s" << std::endl;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Visibility computation complete in " << totalSec << " seconds." << std::endl;
}
