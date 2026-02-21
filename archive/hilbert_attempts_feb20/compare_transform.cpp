#include "nikola/spatial/hilbert_reference.hpp"
#include <iostream>
#include <array>

// Our implementation of PositionToIndex Phase 1 & 2
void our_transform(std::array<uint32_t, 9>& X, uint32_t order) {
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
    
    // Phase 2: Remove gray code
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
    std::array<uint32_t, 9> our_X = {1,0,0,0,0,0,0,0,0};
    std::array<uint8_t, 9> sp_X = {1,0,0,0,0,0,0,0,0};
    
    our_transform(our_X, 2);
    auto sp_result = hilbert::v1::PositionToIndex(sp_X);
    
    std::cout << "Our result (bits 0-1):   ";
    for (auto v : our_X) std::cout << (v & 3) << " ";
    std::cout << "\nSpectral3d (bits 0-1):   ";
    for (auto v : sp_result) std::cout << ((int)v & 3) << " ";
    std::cout << "\n";
    
    bool match = true;
    for (size_t i = 0; i < 9; ++i) {
        if ((our_X[i] & 3) != (sp_result[i] & 3)) {
            match = false;
            break;
        }
    }
    std::cout << (match ? "MATCH!" : "DIFFERENT!") << "\n";
    
    return 0;
}
