#include <iostream>
#include <array>
#include <cstdint>
#include <bitset>
using namespace std;

int main() {
    // Simulate packing for X=[2,0,0,0,0,0,0,0,0]
    array<uint32_t, 9>  X = {2,0,0,0,0,0,0,0,0};
    int order_ = 2;
    const int DIM = 9;
    
    uint64_t index = 0;
    cout << "Packing X=[2,0,0,0,0,0,0,0,0]:\n";
    
    for (int b = order_ - 1; b >= 0; --b) {
        cout << "Bit " << b << ":\n";
        for (size_t i = 0; i < DIM; ++i) {
            index <<= 1;
            uint32_t bit = (X[i] >> b) & 1;
            index |= bit;
            cout << "  dim " << i << ": bit=" << bit << ", index=" << index << "\n";
        }
    }
    
    cout << "\nFinal index: " << index << "\n";
    cout << "Binary: " << bitset<20>(index) << "\n";
    
    return 0;
}
