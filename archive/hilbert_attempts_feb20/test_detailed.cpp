#include "nikola/spatial/hilbert_scanner.hpp"
#include <iostream>
#include <iomanip>

using namespace nikola::spatial;

void test_round_trip(uint32_t order, const HilbertScanner::Coord9D& coords) {
    HilbertScanner scanner(order);
    
    std::cout << "Order " << order << ": [";
    for (size_t i = 0; i < 9; ++i) {
        std::cout << coords[i];
        if (i < 8) std::cout << ",";
    }
    std::cout << "] -> ";
    
    uint64_t idx = scanner.coords_to_index(coords);
    std::cout << "index " << idx << " -> ";
    
    auto recovered = scanner.index_to_coords(idx);
    std::cout << "[";
    for (size_t i = 0; i < 9; ++i) {
        std::cout << recovered[i];
        if (i < 8) std::cout << ",";
    }
    std::cout << "]";
    
    bool ok = (recovered == coords);
    std::cout << (ok ? " OK" : " FAIL") << "\n";
}

int main() {
    // Test order 2 (4096 total points)
    test_round_trip(2, {0,0,0,0,0,0,0,0,0});
    test_round_trip(2, {1,0,0,0,0,0,0,0,0});
    test_round_trip(2, {3,3,3,3,3,3,3,3,3});
    test_round_trip(2, {2,1,3,0,2,1,3,0,2});
    
    // Test order 3  
    test_round_trip(3, {0,0,0,0,0,0,0,0,0});
    test_round_trip(3, {1,0,0,0,0,0,0,0,0});
    test_round_trip(3, {7,7,7,7,7,7,7,7,7});
    test_round_trip(3, {4,6,5,2,3,4,2,3,5});
    
    return 0;
}
