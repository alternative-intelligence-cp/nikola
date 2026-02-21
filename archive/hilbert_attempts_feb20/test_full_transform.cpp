#include "nikola/spatial/hilbert_scanner.hpp"
#include "nikola/spatial/hilbert_reference.hpp"
#include <iostream>

int main() {
    nikola::spatial::HilbertScanner scanner(2);  // order=2
    
    // Test a few coordinates
    std::array<uint32_t, 9> coords[] = {
        {0,0,0,0,0,0,0,0,0},
        {1,0,0,0,0,0,0,0,0},
        {2,0,0,0,0,0,0,0,0},
        {3,0,0,0,0,0,0,0,0}
    };
    
    for (const auto& c : coords) {
        // Our implementation
        auto our_idx = scanner.coords_to_index(c);
        
        // Spectral3d reference - returns untransposed array
        std::array<uint8_t, 9> sp_coords;
        for (size_t i = 0; i < 9; ++i) sp_coords[i] = c[i];
        auto sp_array = hilbert::v1::PositionToIndex(sp_coords);
        
        // Pack spectral3d's result the same way we pack ours
        uint64_t sp_idx = 0;
        for (int b = 1; b >= 0; --b) {  // order=2 means 2 bits
            for (size_t i = 0; i < 9; ++i) {
                sp_idx <<= 1;
                sp_idx |= (sp_array[i] >> b) & 1;
            }
        }
        
        std::cout << "[" << c[0] << ",0,0...] → ";
        std::cout << "our=" << our_idx << ", sp=" << sp_idx;
        std::cout << (our_idx == sp_idx ? " ✓\n" : " ✗\n");
        
        if (our_idx != sp_idx) {
            std::cout << "  Our bits: ";
            for (size_t i = 0; i < 9; ++i) {
                auto back = scanner.index_to_coords(our_idx);
                std::cout << "[" << back[i] << "]";
            }
            std::cout << "\n  SP bits:  ";
            for (int i = 0; i < 9; ++i) {
                std::cout << "[" << ((int)sp_array[i] & 3) << "]";
            }
            std::cout << "\n";
        }
    }
    
    return 0;
}
