#include "hilbert_reference.hpp"
#include <iostream>
#include <array>

int main() {
    // Test 9D with order 2
    constexpr size_t N = 9;
    
    std::array<uint8_t, N> pos{4, 6, 5, 2, 3, 4, 2, 3, 5};
    
    std::cout << "Input position: ";
    for (auto v : pos) std::cout << int(v) << " ";
    std::cout << "\n";
   
    // To index and back
    std::array<uint8_t, N> idx = hilbert::v1::PositionToIndex(pos);
    std::array<uint8_t, N> recovered = hilbert::v1::IndexToPosition(idx);
    
    std::cout << "Recovered:      ";
    for (auto v : recovered) std::cout << int(v) << " ";
    std::cout << "\n";
    
    bool match = true;
    for (size_t i = 0; i < N; ++i) {
        if (pos[i] != recovered[i]) {
            match = false;
            break;
        }
    }
    
    std::cout << "Round-trip: " << (match ? "PASS" : "FAIL") << "\n";
    
    return match ? 0 : 1;
}
