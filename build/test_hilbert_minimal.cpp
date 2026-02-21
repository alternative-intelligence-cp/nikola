/**
 * @file test_hilbert_minimal.cpp
 * @brief Minimal test to debug Hilbert algorithm step-by-step
 */

#include <iostream>
#include <array>
#include <cstdint>

// Simplified 2D Hilbert to understand the algorithm
static constexpr size_t DIM = 2;

inline uint32_t gray_code(uint32_t x) {
    return x ^ (x >> 1);
}

inline uint32_t gray_inverse(uint32_t g) {
    uint32_t x = g;
    for (uint32_t i = 1; i < 32; i <<= 1) {
        x ^= g >> i;
    }
    return x;
}

uint64_t coords_to_index_2d(uint32_t x, uint32_t y, uint32_t order) {
    uint64_t index = 0;
    
    for (int bit = order - 1; bit >= 0; --bit) {
        uint32_t xi = (x >> bit) & 1;
        uint32_t yi = (y >> bit) & 1;
        
        // Combine into 2-bit value (quadrant)
        uint32_t quad = (xi << 1) | yi;
        
        // Rotate/reflect based on quadrant
        // Standard 2D Hilbert curve mapping
        if (yi == 0) {
            if (xi == 1) {
                x = ~x;
                y = ~y;
            }
            std::swap(x, y);
        }
        
        index = (index << 2) | quad;
    }
    
    return index;
}

int main() {
    // Test 2D Hilbert curve (order 2)
    std::cout << "2D Hilbert Curve (order 2, 4x4 grid):\n";
    std::cout << "Index | (x,y) | Expected\n";
    std::cout << "------|-------|----------\n";
    
    // Manually verify first few points
    std::cout << "  0   | (0,0) | (0,0)\n";
    std::cout << "  1   | (1,0) | (1,0)\n";
    std::cout << "  2   | (1,1) | (1,1)\n";
    std::cout << "  3   | (0,1) | (0,1)\n";
    
    std::cout << "\nActual algorithm output:\n";
    for (uint32_t y = 0; y < 4; ++y) {
        for (uint32_t x = 0; x < 4; ++x) {
            uint64_t idx = coords_to_index_2d(x, y, 2);
            std::cout << "(" << x << "," << y << ") → " << idx << "\n";
        }
    }
    
    return 0;
}
