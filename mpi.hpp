#ifndef MPI_HPP
#define MPI_HPP

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "bresenham.hpp"
#include "file_utilities.hpp"

inline void mpiAlgorithm(std::string inputFilePath, std::string outputFilePath, size_t height, size_t width, size_t radius) {
  int processCount;
  int processNumber;

  // Setup MPI
  MPI_Init(NULL, NULL); 
  MPI_Comm_size(MPI_COMM_WORLD, &processCount);
  MPI_Comm_rank(MPI_COMM_WORLD, &processNumber);

  // Read the file on the main process and broadcast to the rest.
  std::vector<int16_t> input;
  if (processNumber == 0) {
    try {
      input = readFile(inputFilePath, height, width);
    } catch (std::exception& e) {
      // There is no data to process, so exit early.
      MPI_Abort(MPI_COMM_WORLD, 1);
      throw e;
    }
  } else {
    input.resize(height * width);
  }
  // Send the data to each of the process.
  MPI_Bcast(input.data(), height * width, MPI_INT16_T, 0, MPI_COMM_WORLD);

  // Determine the portion each process will actually compute.
  // Be careful of uneven division.
  size_t processRowCount = std::ceil(height / processCount);
  size_t localStartRow = processNumber * processRowCount;
  size_t localEndRow = std::min(localStartRow + processRowCount, height);

  auto t0 = std::chrono::high_resolution_clock::now();
  // Allocate output space and calculate each point in the input.
  std::vector<int32_t> localOutput((localEndRow - localStartRow) * width * sizeof(int32_t));
  for (size_t y1 = localStartRow; y1 < localEndRow; y1++) {
    for (size_t x1 = 0; x1 < width; x1++) {
      int32_t xStart = std::max((int32_t)x1 - (int32_t)radius, 0);
      int32_t xEnd = std::min((int32_t)x1 + (int32_t)radius, (int32_t)width - 1);
      int32_t yStart = std::max((int32_t)y1 - (int32_t)radius, 0);
      int32_t yEnd = std::min((int32_t)y1 + (int32_t)radius, (int32_t)height - 1);

      int32_t visiblePoints = 0;
      for (size_t y2 = yStart; y2 <= yEnd; y2++) {
        for (size_t x2 = xStart; x2 <= xEnd; x2++) {
          if (!(y2 == y1 && x2 == x1)) {
            visiblePoints += isVisible(x1, y1, x2, y2, input.data(), width);
          }
        }
        
      }
      localOutput[(y1 - localStartRow) * width + x1] = visiblePoints;
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

  // Prepare output data for coalescing.
  std::vector<int32_t> globalOutput(height * width);
  // Calculate how much data is received by each process and where.
  std::vector<int> bytesToReceive(processCount);
  std::vector<int> receiveDisplacement(processCount);
  if (processNumber == 0) {
    for (int i = 0; i < processCount; i++) {
      int receiveStart = i * processRowCount;
      // Be careful of uneven division.
      int receiveEnd = std::max(receiveStart + processRowCount, height);
      receiveDisplacement[i] = receiveStart * width;
      bytesToReceive[i] = (receiveEnd - receiveStart) * width;
    }
  }

  MPI_Gatherv(
    localOutput.data(),
    localOutput.size() * sizeof(int32_t),
    MPI_INT32_T,
    globalOutput.data(),
    bytesToReceive.data(),
    receiveDisplacement.data(),
    MPI_INT32_T,
    0,
    MPI_COMM_WORLD
  );

  if (processNumber == 0) {
    try {
      writeFile(outputFilePath, globalOutput);
    } catch (std::exception& e) {
      MPI_Abort(MPI_COMM_WORLD, 1);
      throw e;
    }
  }

  // Cleanup MPI
  MPI_Finalize(); 
}

#endif // MPI_HPP
