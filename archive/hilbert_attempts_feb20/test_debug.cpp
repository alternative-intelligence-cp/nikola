#include "nikola/spatial/hilbert_scanner.hpp"
#include <iostream>

using namespace nikola::spatial;

int main() {
    HilbertScanner scanner(3);  // Order 3 (8x8x...x8 grid)
    
    std::array<uint32_t, 9> test_coords = {4, 6, 5, 2, 3, 4, 2, 3, 5};    
    std::cout << "Original coords: ";
    for (auto c : test_coords) std::cout << c << " ";
    std::cout << "\n";
    
    uint64_t index = scanner.coords_to_index(test_coords);
    std::cout << "Computed index: " << index << "\n";
    
    auto recovered = scanner.index_to_coords(index);
    std::cout << "Recovered coords: ";
    for (auto c : recovered) std::cout << c << " ";
    std::cout << "\n";
    
    std::cout << "Match: " << (recovered == test_coords ? "YES" : "NO") << "\n";
    
    return 0;
}
