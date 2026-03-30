#include <cmath>
#include <cstdint>

// TODO: This needs to either return a value or update a global/shared value.
// Bresenham's Line Algorithm: https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
void plotLine(int x1, int y1, int x2, int y2) {
  int dx = abs(x2 - x1);
  int dy = abs(y2 - y1);
  
  // Determine the direction of the line.
  int sx = (x1 < x2) ? 1 : -1;
  int sy = (y1 < y2) ? 1 : -1;
  
  // Current evaluation point, initialized point to first point argument.
  int x = x1;
  int y = y1;

  if (dx > dy) {
    // Initial decision parameter.
    int parameter = 2 * dy - dx; 
  
    while (x <= dx) {
      if (parameter < 0) {
          parameter += 2 * dy;
      } else {
          parameter += 2 * (dy - dx);
          y += sy;
      }
      x += sx;
    }
  } else {
    // Initial decision parameter.
    int parameter = 2 * dx - dy; 

    while (y <= dy) {
    if (parameter < 0) {
      parameter += 2 * dx;
    } else {
      x += sx;
      parameter += 2 * (dx - dy);
    }
    y += sy;
  }
}
