/**
 * @file wallet.cpp
 * @brief Phase 131 — SimulatedWallet implementation
 */

#include "nikola/economy/wallet.hpp"

#include <sstream>
#include <iomanip>
#include <algorithm>
#include <stdexcept>

namespace nikola::economy {

// ---------------------------------------------------------------------------
// Static helpers
// ---------------------------------------------------------------------------

std::string SimulatedWallet::derive_address(const std::string& seed) {
    // FNV-1a over seed → 160-bit-equivalent hex (20 bytes = 40 chars)
    // We compute two 64-bit hashes with different offsets to get 128 bits
    // then pad to 40 hex chars (160 bits).
    constexpr uint64_t FNV_PRIME   = 0x100000001b3ULL;
    constexpr uint64_t FNV_OFFSET1 = 0xcbf29ce484222325ULL;
    constexpr uint64_t FNV_OFFSET2 = 0x14650fb0739d0383ULL;

    uint64_t h1 = FNV_OFFSET1;
    uint64_t h2 = FNV_OFFSET2;
    for (char c : seed) {
        h1 ^= static_cast<uint8_t>(c); h1 *= FNV_PRIME;
        h2 ^= static_cast<uint8_t>(~c & 0xff); h2 *= FNV_PRIME;
    }

    std::ostringstream oss;
    oss << "0x"
        << std::hex << std::setw(16) << std::setfill('0') << h1
        << std::hex << std::setw(8)  << std::setfill('0')
        << static_cast<uint32_t>(h2 >> 32);
    return oss.str();
}

std::string SimulatedWallet::build_expected_sig(const std::string& address,
                                                  const std::string& data) {
    const std::string addr_part = address.size() >= 8 ? address.substr(0, 8)
                                                       : address;
    const std::string data_part = data.size() >= 8 ? data.substr(0, 8) : data;
    return "sig_" + addr_part + "_" + data_part;
}

// ---------------------------------------------------------------------------
// NeuralWallet interface
// ---------------------------------------------------------------------------

std::string SimulatedWallet::derive_identity(const std::string& seed) {
    address_     = derive_address(seed);
    private_key_ = "privkey_" + seed.substr(0, std::min<size_t>(16, seed.size()));
    return private_key_;
}

std::string SimulatedWallet::get_address() const {
    return address_;
}

std::string SimulatedWallet::sign(const std::string& data) {
    ++sign_count_;
    return build_expected_sig(address_, data);
}

bool SimulatedWallet::verify(const std::string& data,
                              const std::string& signature,
                              const std::string& address) {
    const std::string expected = build_expected_sig(address, data);
    if (signature == expected) {
        ++verify_count_;
        return true;
    }
    return false;
}

uint64_t SimulatedWallet::get_balance_wei() const {
    return balance_wei_;
}

void SimulatedWallet::credit(uint64_t amount_wei) {
    balance_wei_ += amount_wei;
}

bool SimulatedWallet::debit(uint64_t amount_wei) {
    if (amount_wei > balance_wei_) return false;
    balance_wei_ -= amount_wei;
    ++debit_count_;
    return true;
}

// ---------------------------------------------------------------------------
// SimulatedWallet extras
// ---------------------------------------------------------------------------

bool SimulatedWallet::has_identity() const {
    return !address_.empty();
}

size_t SimulatedWallet::sign_count()   const { return sign_count_;   }
size_t SimulatedWallet::verify_count() const { return verify_count_; }
size_t SimulatedWallet::debit_count()  const { return debit_count_;  }

} // namespace nikola::economy
