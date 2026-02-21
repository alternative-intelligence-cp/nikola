#include "nikola/spatial/hilbert_scanner.hpp"
#include "nikola/spatial/hilbert_reference.hpp"
#include <iostream>

uint64_t coords_to_index_spectral(const std::array<uint32_t, 9>& coords, uint32_t order) {
    std::array<uint8_t, 9> pos;
    for (size_t i = 0; i <  9; ++i) {
        pos[i] = static_cast<uint8_t>(coords[i]);
    }
    
    auto idx = hilbert::v1::PositionToIndex(pos);
    
    uint64_t result = 0;
    for (int b = order - 1; b >= 0; --b) {
        for (size_t i = 0; i < 9; ++i) {
            uint64_t bit = (idx[i] >> b) & 1U;
            result = (result << 1) | bit;
        }
    }
    
    return result;
}

int main() {
    using namespace nikola::spatial;
    
    HilbertScanner scanner(2);
    
    // Test a few simple cases
   std::array<uint32_t, 9> test_coords[] = {
        {0,0,0,0,0,0,0,0,0},
        {1,0,0,0,0,0,0,0,0},
        {2,0,0,0,0,0,0,0,0},
        {3,0,0,0,0,0,0,0,0},
        {0,1,0,0,0,0,0,0,0},
        {1,1,0,0,0,0,0,0,0},
    };
    
    for (auto& coords : test_coords) {
        uint64_t my_idx = scanner.coords_to_index(coords);
        uint64_t sp_idx = coords_to_index_spectral(coords, 2);
        
        std::cout << "[";
        for (int i = 0; i < 9; ++i) {
            std::cout << coords[i];
            if (i < 8) std::cout << ",";
        }
        std::cout << "] -> mine:" << my_idx << " spectral:" << sp_idx;
        if (my_idx != sp_idx) std::cout << " DIFF!";
        std::cout << "\n";
    }
    
    return 0;
}
