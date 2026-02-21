#include "hilbert_reference.hpp"
#include <iostream>
#include <array>
#include <cstdint>

int main() {
    // Test 3D with order 2 (2 bits per dimension)
    constexpr size_t N = 3;
    std::array<uint8_t, N> pos{1, 2, 3};
    
    std::cout << "Input position: (" << int(pos[0]) << "," << int(pos[1]) << "," << int(pos[2]) << ")\n";
    
    // Convert to index
    std::array<uint8_t, N> idx = hilbert::v1::PositionToIndex(pos);
    std::cout << "Index array: [" << int(idx[0]) << "," << int(idx[1]) << "," << int(idx[2]) << "]\n";
    
    // Convert back to position
    std::array<uint8_t, N> recovered = hilbert::v1::IndexToPosition(idx);
    std::cout << "Recovered position: (" << int(recovered[0]) << "," << int(recovered[1]) << "," << int(recovered[2]) << ")\n";
    
    // Round-trip test
    bool match = (recovered[0] == pos[0] && recovered[1] == pos[1] && recovered[2] == pos[2]);
    std::cout << "Round-trip: " << (match ? "PASS" : "FAIL") << "\n";
    
    return match ? 0 : 1;
}
