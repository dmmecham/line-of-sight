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

__host__ __device__ bool isVisible2(int16_t x0, int16_t y0, int16_t x1, int16_t y1, const int16_t* elevations, int16_t width)
{
  int dx = abs(x1 - x0);
  int dy = abs(y1 - y0);

  int sx = (x0 < x1) ? 1 : -1;
  int sy = (y0 < y1) ? 1 : -1;

  int err = dx - dy;

  int16_t base = elevations[y0 * width + x0];

  float max_slope = -1e30f;
  bool max_positive = false;
  
  int x = x0, y = y0;

  while (!(x == x1 && y == y1)) {
    int e2 = 2 * err;

    if (e2 > -dy) { err -= dy; x += sx; }
    if (e2 < dx)  { err += dx; y += sy; }

    float dx0 = float(x - x0);
    float dy0 = float(y - y0);

    int dist2 = dx0 * dx0 + dy0 * dy0;
    if (dist2 == 0) continue;
    int dz = elevations[y * width + x] - base;
    float slope = float(dz * dz) / float(dist2);
    bool positive = (dz >= 0);

    bool visible = false;

    if (max_slope < 0.0f) {
      visible = true;
    }
    else if (positive && !max_positive) {
      // Any positive slope beats previous negative
      visible = true;
    }
    else if (positive == max_positive) {
      if (slope > max_slope)
        visible = true;
    }

    if (visible) {
      max_slope = slope;
      max_positive = positive;
    } else {
      return false;
    }
  }

  return true;
}
