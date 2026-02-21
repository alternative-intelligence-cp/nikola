#include "nikola/spatial/hilbert_reference.hpp"
#include <iostream>
#include <array>
#include <cstdint>

void my_position_to_index(std::array<uint32_t, 9>& X, uint32_t order) {
    std::cout << "After input: ";
    for (auto v : X) std::cout << v << " ";
    std::cout << "\n";
    
    // Phase 1: Reverse transforms
    {
        uint32_t cur_bit = 1U << (order - 1);
        uint32_t low_bits;
        
        do {
            low_bits = cur_bit - 1;
            
            for (size_t n = 0; n < 9; ++n) {
                if (X[n] & cur_bit) {
                    X[0] ^= low_bits;
                } else {
                    uint32_t t = (X[n] ^ X[0]) & low_bits;
                    X[n] ^= t;
                    X[0] ^= t;
                }
            }
            
            cur_bit >>= 1;
        } while (low_bits > 1);
    }
    
    std::cout << "After phase 1: ";
    for (auto v : X) std::cout << v << " ";
    std::cout << "\n";
    
    // Phase 2: Remove gray code
    {
        for (size_t n = 1; n < 9; ++n) {
            X[n] ^= X[n-1];
        }
        
        std::cout << "After XOR chain: ";
        for (auto v : X) std::cout << v << " ";
        std::cout << "\n";
        
        uint32_t t = 0;
        uint32_t cur_bit = 1U << (order - 1);
        do {
            if (X[8] & cur_bit) {
                t ^= cur_bit;
            }
            cur_bit >>= 1;
        } while (cur_bit);
        
        std::cout << "t accumulated: " << t << "\n";
        X[0] ^= t;
    }
    
    std::cout << "Final transpose: ";
    for (auto v : X) std::cout << v << " ";
    std::cout << "\n";
}

int main() {
    std::array<uint32_t, 9> my_X = {1,0,0,0,0,0,0,0,0};
    my_position_to_index(my_X, 3);
    
    std::cout << "\nSpectral3d for comparison:\n";
    std::array<uint8_t, 9> sp_X = {1,0,0,0,0,0,0,0,0};
    auto sp_trans = hilbert::v1::PositionToIndex(sp_X);
    std::cout << "Spectral3d (bits 0-2): ";
    for (auto v : sp_trans) std::cout << ((int)v & 7) << " ";
    std::cout << "\n";
    
    return 0;
}
