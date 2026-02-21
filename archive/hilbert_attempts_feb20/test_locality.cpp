#include "nikola/spatial/hilbert_scanner.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace nikola::spatial;

int main() {
    HilbertScanner scanner(2);
    
    // Test a few specific adjacent pairs
    std::array<uint32_t, 9> test_cases[] = {
        {0,0,0,0,0,0,0,0,0},
        {1,0,0,0,0,0,0,0,0},
        {2,0,0,0,0,0,0,0,0},
        {3,0,0,0,0,0,0,0,0}
    };
    
    std::cout << "Testing sequential coordinates in dimension 0:\n";
    for (auto& coords : test_cases) {
        uint64_t idx = scanner.coords_to_index(coords);
        std::cout << "  [" << coords[0] << ",0,0,0,0,0,0,0,0] -> index " << idx;
        
        // Check distance to next if not last
        if (coords[0] < 3) {
            auto next_coords = coords;
            next_coords[0]++;
            uint64_t next_idx = scanner.coords_to_index(next_coords);
            int64_t dist = std::abs(static_cast<int64_t>(next_idx - idx));
            std::cout << ", distance to next: " << dist;
            if (dist < 26214) std::cout << " OK";
            else std::cout << " FAIL (too far!)";
        }
        std::cout << "\n";
    }
    
    std::cout << "\nRandom scattered cases:\n";
    srand(42);
    for (int i = 0; i < 5; ++i) {
        std::array<uint32_t, 9> coords;
        for (int d = 0; d < 9; ++d) {
            coords[d] = rand() % 4;
        }
        
        std::cout << "  [";
        for (int d = 0; d < 9; ++d) {
            std::cout << coords[d];
            if (d < 8) std::cout << ",";
        }
        std::cout << "] -> " << scanner.coords_to_index(coords);
        
        if (coords[0] < 3) {
            coords[0]++;
            uint64_t adj_idx = scanner.coords_to_index(coords);
            coords[0]--;
            uint64_t base_idx = scanner.coords_to_index(coords);
            int64_t dist = std::abs(static_cast<int64_t>(adj_idx - base_idx));
            std::cout << ", dist: " << dist;
            if (dist < 26214) std::cout << " OK";
            else std::cout << " FAIL";
        }
        std::cout << "\n";
    }
    
    return 0;
}
