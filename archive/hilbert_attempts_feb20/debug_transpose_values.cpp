#include "nikola/spatial/hilbert_reference.hpp"
#include <iostream>
#include <iomanip>
#include <array>
#include <cstdint>

void my_position_to_index(std::array<uint32_t, 9>& X, uint32_t order) {
    // Phase 1
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
    
    // Phase 2
    {
        for (size_t n = 1; n < 9; ++n) {
            X[n] ^= X[n-1];
        }
        
        uint32_t cur_bit = 1U << (order - 1);
        uint32_t t = 0;
        
        do {
            if (X[8] & cur_bit) {
                t ^= (cur_bit - 1);
            }
            cur_bit >>= 1;
        } while (cur_bit > 1);
        
        for (auto& v : X) {
            v ^= t;
        }
    }
}

int main() {
    std::array<uint32_t, 9> my_X = {1,0,0,0,0,0,0,0,0};
    std::array<uint8_t, 9> sp_X = {1,0,0,0,0,0,0,0,0};
    
    my_position_to_index(my_X, 2);
    auto sp_trans = hilbert::v1::PositionToIndex(sp_X);
    
    std::cout << "My  transpose (bits 0-1): ";
    for (auto v : my_X) std::cout << (v & 3) << " ";
    std::cout << "\n";
    
    std::cout << "Spectral3d output (bits 0-1): ";
    for (auto v : sp_trans) std::cout << ((int)v & 3) << " ";
    std::cout << "\n";
    
    std::cout << "\nSpectral3d full output (hex): ";
    for (auto v : sp_trans) std::cout << std::hex << std::setfill('0') << std::setw(2) << (int)v << " ";
    std::cout << std::dec << "\n";
    
    return 0;
}
