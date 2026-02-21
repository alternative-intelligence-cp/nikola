#include <iostream>
#include <array>
#include <cstdint>

int main() {
    // Test the packing/unpacking with a known pattern
    constexpr size_t DIM = 9;
    constexpr uint32_t ORDER = 2;
    
    std::array<uint32_t, DIM> test{0, 1, 2, 3, 0, 1, 2, 3, 0};
    
    std::cout << "Original: ";
    for (auto v : test) std::cout << v << " ";
    std::cout << "\n";
    
    // Pack (lexographic: MSB first)
    uint64_t packed = 0;
    for (size_t i = 0; i < DIM; ++i) {
        packed |= (uint64_t(test[i]) << (ORDER * (DIM - 1 - i)));
    }
    
    std::cout << "Packed: 0x" << std::hex << packed << std::dec << "\n";
    
    // Unpack
    std::array<uint32_t, DIM> unpacked{};
    for (size_t i = 0; i < DIM; ++i) {
        unpacked[i] = (packed >> (ORDER * (DIM - 1 - i))) & ((1U << ORDER) - 1);
    }
    
    std::cout << "Unpacked: ";
    for (auto v : unpacked) std::cout << v << " ";
    std::cout << "\n";
    
    // Check
    bool match = true;
    for (size_t i = 0; i < DIM; ++i) {
        if (test[i] != unpacked[i]) {
            std::cout << "MISMATCH at index " << i << ": " << test[i] << " != " << unpacked[i] << "\n";
            match = false;
        }
    }
    
    std::cout << "Pack/Unpack: " << (match ? "PASS" : "FAIL") << "\n";
    
    return match ? 0 : 1;
}
