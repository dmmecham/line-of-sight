#ifndef NO_THREADS_HPP
#define NO_THREADS_HPP

#include <chrono>
#include <iostream>
#include <vector>

#include "bresenham.hpp"
#include "file_utilities.hpp"

// Serial implementation
inline void serialAlgorithm(std::string inputFilePath, std::string outputFilePath, size_t height, size_t width, size_t radius) {
    std::vector<int16_t> heightMap = readFile(inputFilePath, height, width);
    std::vector<int32_t> visibleCounts = std::vector<int32_t>(heightMap.size(), 0);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (size_t y1 = 0; y1 < height; ++y1) {
        for (size_t x1 = 0; x1 < width; ++x1) {
          size_t idx1 = y1 * width + x1;
            int32_t count = 0;

            // Radius-limited window
            int32_t y0 = std::max(0, (int32_t)y1 - (int32_t)radius);
            int32_t yN = std::min((int32_t)height - 1, (int32_t)y1 + (int32_t)radius);
            int32_t x0 = std::max(0, (int32_t)x1 - (int32_t)radius);
            int32_t xN = std::min((int32_t)width - 1, (int32_t)x1 + (int32_t)radius);

            for (size_t y2 = y0; y2 <= yN; ++y2) {
                for (size_t x2 = x0; x2 <= xN; ++x2) {
                    // Skip self-comparison
                    if (x1 == x2 && y1 == y2) continue;
                    // Check line of sight using Bresenham's algorithm
                    if (isVisible(x1, y1, x2, y2, heightMap.data(), width)) ++count;
                }
            }

            visibleCounts[idx1] = count;
        }

        // Print progress every 100 rows
        if ((y1 % 100) == 0) {
            auto tnow = std::chrono::high_resolution_clock::now();
            double sec = std::chrono::duration<double>(tnow - t0).count();
            std::cout << "Row " << y1 << " elapsed: " << sec << "s" << std::endl;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double totalSec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Visibility computation complete in " << totalSec << " seconds." << std::endl;

    //for (size_t i = 0; i < visibleCounts.size(); ++i) {
    //  std::cout << "Point (" << (i % width) << ", " << (i / width) << ") sees " << visibleCounts[i] << " points." << std::endl;
    //}

    writeFile(outputFilePath, visibleCounts);
}

#endif // NO_THREADS_HPP