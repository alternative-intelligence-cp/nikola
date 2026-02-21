#include "nikola/spatial/hilbert_scanner.hpp"
#include <iostream>
#include <array>

int main() {
    nikola::spatial::HilbertScanner scanner(2);
    
    // Test [1,0,0,0,0,0,0,0,0]
    std::array<uint32_t, 9> c = {1,0,0,0,0,0,0,0,0};
    auto idx = scanner.coords_to_index(c);
    auto back = scanner.index_to_coords(idx);
    
    std::cout << "Input:  [" << c[0];
    for (size_t i = 1; i < 9; ++i) std::cout << "," << c[i];
    std::cout << "]\n";
    
    std::cout << "Index:  " << idx << " (0x" << std::hex << idx << std::dec << ")\n";
    
    std::cout << "Output: [" << back[0];
    for (size_t i = 1; i < 9; ++i) std::cout << "," << back[i];
    std::cout << "]\n";
    
    bool match = true;
    for (size_t i = 0; i < 9; ++i) {
        if (c[i] != back[i]) {
            match = false;
            break;
        }
    }
    std::cout << "Round-trip: " << (match ? "PASS" : "FAIL") << "\n";
    
    return 0;
}
