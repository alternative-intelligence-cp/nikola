#pragma once
/**
 * @file include/nikola/security/io_guard.hpp
 * @brief v0.3.6 QoL: IOGuard token-bucket I/O rate limiter.
 *
 * Design target from integration notes:
 *   - Refill rate: 1 MiB/s
 *   - Burst capacity: 256 KiB
 *
 * The implementation is thread-safe and deterministic-test friendly via
 * overloads that accept an explicit timestamp.
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace nikola::security {

inline constexpr std::size_t IOGUARD_DEFAULT_REFILL_BYTES_PER_SEC = 1U * 1024U * 1024U;  // 1 MiB/s
inline constexpr std::size_t IOGUARD_DEFAULT_BURST_BYTES          = 256U * 1024U;         // 256 KiB

class IOGuard {
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    explicit IOGuard(std::size_t refill_bytes_per_sec = IOGUARD_DEFAULT_REFILL_BYTES_PER_SEC,
                     std::size_t burst_bytes = IOGUARD_DEFAULT_BURST_BYTES)
        : refill_bytes_per_sec_(refill_bytes_per_sec > 0 ? refill_bytes_per_sec : 1),
          burst_bytes_(burst_bytes > 0 ? burst_bytes : 1),
          tokens_(static_cast<double>(burst_bytes_)),
          last_refill_(Clock::now())
    {}

    [[nodiscard]] bool allow(std::size_t bytes) {
        return allow(bytes, Clock::now());
    }

    [[nodiscard]] bool allow(std::size_t bytes, TimePoint now) {
        std::lock_guard<std::mutex> lock(mutex_);
        refill_to_(now);
        if (bytes == 0) return true;
        if (bytes > burst_bytes_) return false;
        if (tokens_ + 1e-9 < static_cast<double>(bytes)) return false;
        tokens_ -= static_cast<double>(bytes);
        return true;
    }

    [[nodiscard]] std::size_t available_tokens(TimePoint now = Clock::now()) const {
        std::lock_guard<std::mutex> lock(mutex_);
        refill_to_(now);
        return static_cast<std::size_t>(tokens_);
    }

    void set_limits(std::size_t refill_bytes_per_sec,
                    std::size_t burst_bytes,
                    TimePoint now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        refill_to_(now);

        refill_bytes_per_sec_ = refill_bytes_per_sec > 0 ? refill_bytes_per_sec : 1;
        burst_bytes_          = burst_bytes > 0 ? burst_bytes : 1;
        tokens_ = std::min(tokens_, static_cast<double>(burst_bytes_));
    }

    [[nodiscard]] std::chrono::milliseconds time_until_available(
        std::size_t bytes, TimePoint now = Clock::now()) const {
        std::lock_guard<std::mutex> lock(mutex_);
        refill_to_(now);

        if (bytes == 0 || bytes <= static_cast<std::size_t>(tokens_)) {
            return std::chrono::milliseconds{0};
        }
        if (bytes > burst_bytes_) {
            return std::chrono::milliseconds::max();
        }

        const double deficit = static_cast<double>(bytes) - tokens_;
        const double seconds = deficit / static_cast<double>(refill_bytes_per_sec_);
        const auto ms = static_cast<std::int64_t>(seconds * 1000.0 + 0.999); // ceil
        return std::chrono::milliseconds{ms};
    }

    [[nodiscard]] std::size_t refill_rate_bytes_per_sec() const noexcept {
        return refill_bytes_per_sec_;
    }

    [[nodiscard]] std::size_t burst_bytes() const noexcept {
        return burst_bytes_;
    }

private:
    void refill_to_(TimePoint now) const {
        if (now <= last_refill_) return;

        const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now - last_refill_).count();
        if (elapsed_us <= 0) return;

        const double refill = (static_cast<double>(elapsed_us) / 1'000'000.0) *
                              static_cast<double>(refill_bytes_per_sec_);
        tokens_ = std::min(static_cast<double>(burst_bytes_), tokens_ + refill);
        last_refill_ = now;
    }

    std::size_t refill_bytes_per_sec_;
    std::size_t burst_bytes_;

    mutable std::mutex mutex_;
    mutable double     tokens_;
    mutable TimePoint  last_refill_;
};

} // namespace nikola::security
