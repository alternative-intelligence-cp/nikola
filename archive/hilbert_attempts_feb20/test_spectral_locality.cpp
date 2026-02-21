#include "nikola/spatial/hilbert_reference.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <array>

uint64_t coords_to_index_spectral(const std::array<uint32_t, 9>& coords, uint32_t order) {
    std::array<uint8_t, 9> pos;
    for (size_t i = 0; i < 9; ++i) {
        pos[i] = static_cast<uint8_t>(coords[i]);
    }
    
    auto idx = hilbert::v1::PositionToIndex(pos);
    
    uint64_t result = 0;
    for (int b = order - 1; b >= 0; --b) {
        for (size_t i = 0; i < 9; ++i) {
            uint64_t bit = (idx[i] >> b) & 1U;
            result = (result << 1) | bit;
        }
    }
    
    return result;
}

int main() {
    uint32_t order = 2;
    uint32_t bins = 4;
    
    int adjacent_close_count = 0;
    int total_adjacent_pairs = 0;
    
    srand(42);
    
    for (int sample = 0; sample < 100; ++sample) {
        std::array<uint32_t, 9> coords;
        for (int d = 0; d < 9; ++d) {
            coords[d] = rand() % bins;
        }
        
        uint64_t base_index = coords_to_index_spectral(coords, order);
        
        if (coords[0] < bins - 1) {
            coords[0]++;
            uint64_t adj_index = coords_to_index_spectral(coords, order);
            coords[0]--;
            
            uint64_t index_distance = std::abs(static_cast<int64_t>(base_index - adj_index));
            total_adjacent_pairs++;
            
            if (index_distance < 26214) {
                adjacent_close_count++;
            }
        }
    }
    
    double locality_ratio = static_cast<double>(adjacent_close_count) / total_adjacent_pairs;
    std::cout << "Spectral3d locality: " << adjacent_close_count << "/" << total_adjacent_pairs;
    std::cout << " = " << (locality_ratio * 100) << "%\n";
    std::cout << (locality_ratio > 0.85 ? "PASS" : "FAIL") << "\n";
    
    return 0;
}
