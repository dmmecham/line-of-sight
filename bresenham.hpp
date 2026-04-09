#include <cmath>
#include <cstdint>
#include <algorithm>

// Added eye/target height offsets (default to 0 if you just want ground-to-ground)
bool isVisible(int x1, int y1, int x2, int y2, const short* heightMap, int width) {
    // Fetch start and end heights from the map and add offsets
    short h1 = heightMap[y1 * width + x1];
    short h2 = heightMap[y2 * width + x2];

    int dx = abs(x2 - x1);
    int dy = -abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx + dy;

    int steps = std::max(dx, abs(dy));
    if (steps == 0) return true;

    // We use float here so we can increment by fractional heights.
    float heightStep = static_cast<float>(h2 - h1) / steps;
    float currentLineHeight = static_cast<float>(h1);

    int x = x1;
    int y = y1;

    while (true) {
        // Skip the very first and last points to avoid checking 
        // the ground the entities are standing on.
        if ((x != x1 || y != y1) && (x != x2 || y != y2)) {
            if ((float)heightMap[y * width + x] > currentLineHeight) {
                return false; 
            }
        }

        if (x == x2 && y == y2) break;

        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x += sx; }
        if (e2 <= dx) { err += dx; y += sy; }
        
        currentLineHeight += heightStep;
    }

    return true;
}
