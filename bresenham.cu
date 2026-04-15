#include <algorithm>
#include <cmath>
#include <cstdint>

#include "bresenham.hpp"

// Added eye/target height offsets (default to 0 if you just want ground-to-ground)
__host__ __device__ bool isVisible(int16_t x1, int16_t y1, int16_t x2, int16_t y2, const int16_t* heightMap, int16_t width) {
    // Fetch start and end heights from the map and add offsets
    int16_t h1 = heightMap[y1 * width + x1];
    int16_t h2 = heightMap[y2 * width + x2];


    int32_t dx = abs(x2 - x1);
    int32_t dy = -abs(y2 - y1);
    int32_t sx = (x1 < x2) ? 1 : -1;
    int32_t sy = (y1 < y2) ? 1 : -1;
    int32_t err = dx + dy;

    int32_t steps = std::max(dx, abs(dy));
    if (steps == 0) return true;

    // We use float here so we can increment by fractional heights.
    float heightStep = static_cast<float>(h2 - h1) / steps;
    float currentLineHeight = static_cast<float>(h1);

    int32_t x = x1;
    int32_t y = y1;

    while (true) {
        // Skip the very first and last points to avoid checking 
        // the ground the entities are standing on.
        if ((x != x1 || y != y1) && (x != x2 || y != y2)) {
            if ((float)heightMap[y * width + x] > currentLineHeight) {
                return false; 
            }
        }

        if (x == x2 && y == y2) break;

        int32_t e2 = 2 * err;
        if (e2 >= dy) { err += dy; x += sx; }
        if (e2 <= dx) { err += dx; y += sy; }
        
        currentLineHeight += heightStep;
    }

    return true;
}
