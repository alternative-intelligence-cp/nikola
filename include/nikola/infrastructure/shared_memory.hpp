/**
 * @file include/nikola/infrastructure/shared_memory.hpp
 * @brief RAII Shared Memory + Lock-Free Seqlock IPC for Nikola physics stream.
 *
 * Resolves Gap 4.3: Shared Memory Lifecycle Management.
 *
 * High-level design:
 *   WaveformSHM    — RAII shm_open/ftruncate/mmap/munmap/shm_unlink lifecycle.
 *   SeqlockFrame<T>— Lock-free single-writer / multi-reader IPC buffer,
 *                    zero-copy at 60 fps physics throughput.
 *   SeqlockWriter<T>  — Writer side: sequence stamp + memcpy frame.
 *   SeqlockReader<T>  — Reader side: retry loop guarded by sequence invariant.
 *   cleanup_stale_shm()  — Boot-time sweep of /dev/shm/nikola_* leftovers.
 *
 * Concurrency guarantee (seqlock):
 *   - sequence even  → stable (safe to read)
 *   - sequence odd   → write in progress (readers spin)
 *   - readers retry while seq before ≠ seq after
 *
 * Performance:
 *   - Zero serialization: raw struct copy into shared mapping.
 *   - Writer never blocks.
 *   - Readers spin only during the ~µs write window (single memcpy).
 *
 * Max total SHM (safety cap): 16 GiB.
 */

#pragma once

#include <atomic>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>

// POSIX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nikola::infrastructure {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

inline constexpr std::size_t SHM_MAX_TOTAL_BYTES = 16ULL * 1024 * 1024 * 1024; ///< 16 GiB cap
inline constexpr std::string_view SHM_PREFIX     = "/nikola_";                  ///< Segment name prefix

// ---------------------------------------------------------------------------
// WaveformSHM  (Gap 4.3 – RAII lifecycle)
// ---------------------------------------------------------------------------

/**
 * @class WaveformSHM
 * @brief POSIX shared-memory segment with RAII create/map/destroy.
 *
 * Usage:
 *   WaveformSHM shm("/nikola_physics", sizeof(PhysicsFrame));
 *   auto* frame = static_cast<PhysicsFrame*>(shm.data());
 *   // shm destroyed ↓ automatically calls munmap + shm_unlink
 */
class WaveformSHM {
public:
    /**
     * @param segment_name  POSIX name, e.g. "/nikola_physics" (must start with /).
     * @param bytes         Byte size to allocate.  Must be > 0 and ≤ SHM_MAX_TOTAL_BYTES.
     */
    WaveformSHM(const std::string& segment_name, std::size_t bytes)
        : name_(segment_name), size_(bytes)
    {
        if (bytes == 0 || bytes > SHM_MAX_TOTAL_BYTES) {
            throw std::runtime_error("WaveformSHM: invalid size " + std::to_string(bytes));
        }

        // 1. Create / open
        fd_ = ::shm_open(name_.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd_ == -1) {
            throw std::runtime_error("WaveformSHM: shm_open failed for " + name_);
        }

        // 2. Set size
        if (::ftruncate(fd_, static_cast<off_t>(size_)) == -1) {
            ::close(fd_);
            ::shm_unlink(name_.c_str());
            throw std::runtime_error("WaveformSHM: ftruncate failed (size limit exceeded?)");
        }

        // 3. Map
        ptr_ = ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) {
            ptr_ = nullptr;
            ::close(fd_);
            ::shm_unlink(name_.c_str());
            throw std::runtime_error("WaveformSHM: mmap failed for " + name_);
        }
    }

    /// Move-construct transfers ownership.
    WaveformSHM(WaveformSHM&& o) noexcept
        : name_(std::move(o.name_)), fd_(o.fd_), ptr_(o.ptr_), size_(o.size_)
    {
        o.fd_   = -1;
        o.ptr_  = nullptr;
        o.size_ = 0;
    }

    /// Move-assign (transfers ownership).
    WaveformSHM& operator=(WaveformSHM&& o) noexcept {
        if (this != &o) {
            // Destroy existing
            if (ptr_)       ::munmap(ptr_, size_);
            if (fd_ != -1)  ::close(fd_);
            if (!name_.empty()) ::shm_unlink(name_.c_str());
            // Transfer
            name_ = std::move(o.name_);
            fd_   = o.fd_;
            ptr_  = o.ptr_;
            size_ = o.size_;
            o.fd_   = -1;
            o.ptr_  = nullptr;
            o.size_ = 0;
        }
        return *this;
    }
    WaveformSHM(const WaveformSHM&)       = delete;
    WaveformSHM& operator=(const WaveformSHM&) = delete;

    ~WaveformSHM() {
        if (ptr_)  ::munmap(ptr_, size_);
        if (fd_ != -1) ::close(fd_);
        if (!name_.empty()) ::shm_unlink(name_.c_str());
    }

    [[nodiscard]] void*       data()         noexcept { return ptr_; }
    [[nodiscard]] const void* data()   const noexcept { return ptr_; }
    [[nodiscard]] std::size_t get_size() const noexcept { return size_; }
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] bool        valid()  const noexcept { return ptr_ != nullptr; }

private:
    std::string name_;
    int         fd_   = -1;
    void*       ptr_  = nullptr;
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// SeqlockFrame<T>  — shared-memory layout
// ---------------------------------------------------------------------------

/**
 * @struct SeqlockFrame
 * @brief Lock-free shared-memory frame header + payload.
 *
 * The sequence counter protocol:
 *   odd  = write in progress
 *   even = stable data
 *
 * Writer must do:
 *   seq.fetch_add(1)   // marks odd
 *   <write payload>
 *   seq.fetch_add(1)   // marks even
 *
 * Readers: read seq_before (even), read payload, read seq_after;
 *          retry while seq_before != seq_after or seq_before is odd.
 *
 * @tparam T  Plain frame type (must be trivially-copyable).
 */
template<typename T>
struct SeqlockFrame {
    static_assert(std::is_trivially_copyable_v<T>,
        "SeqlockFrame payload T must be trivially copyable");

    std::atomic<uint64_t> sequence{0};   ///< Even = stable, odd = writing
    uint64_t              timestamp_ns{0};
    uint32_t              frame_number{0};
    uint32_t              _pad{0};       ///< Alignment
    T                     payload{};

    /// Total size of this struct — useful for ftruncate.
    static constexpr std::size_t byte_size() noexcept { return sizeof(SeqlockFrame<T>); }
};

// ---------------------------------------------------------------------------
// SeqlockWriter<T>
// ---------------------------------------------------------------------------

/**
 * @class SeqlockWriter
 * @brief Owns the write side of a SeqlockFrame in shared memory.
 */
template<typename T>
class SeqlockWriter {
public:
    explicit SeqlockWriter(SeqlockFrame<T>* frame) noexcept
        : frame_(frame)
    {}

    /**
     * @brief Atomically publish a new frame into shared memory.
     * @param data  New payload to write.
     */
    void write(const T& data) noexcept {
        // Mark as writing (odd)
        frame_->sequence.fetch_add(1, std::memory_order_acq_rel);

        // Write payload
        frame_->payload       = data;
        frame_->timestamp_ns  = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count()
        );
        ++frame_->frame_number;

        // Mark as stable (even)
        frame_->sequence.fetch_add(1, std::memory_order_acq_rel);
    }

    [[nodiscard]] uint32_t frame_number() const noexcept { return frame_->frame_number; }

private:
    SeqlockFrame<T>* frame_;
};

// ---------------------------------------------------------------------------
// SeqlockReader<T>
// ---------------------------------------------------------------------------

/**
 * @class SeqlockReader
 * @brief Read side of a SeqlockFrame.  Non-blocking; retries on concurrent write.
 *
 * For a 60 Hz physics writer the spin window is < 1 µs per frame.
 */
template<typename T>
class SeqlockReader {
public:
    explicit SeqlockReader(const SeqlockFrame<T>* frame) noexcept
        : frame_(frame)
    {}

    /**
     * @brief Read the latest stable frame.
     * @param out  Destination for payload copy.
     * @return true (always; may spin briefly).
     */
    bool read(T& out) const noexcept {
        uint64_t seq1, seq2;
        do {
            seq1 = frame_->sequence.load(std::memory_order_acquire);
            if (seq1 & 1u) continue; // Write in progress — spin

            // Snapshot payload
            out = frame_->payload;

            // Memory fence then re-check sequence
            seq2 = frame_->sequence.load(std::memory_order_acquire);
        } while (seq1 != seq2);

        return true;
    }

    [[nodiscard]] uint64_t current_sequence() const noexcept {
        return frame_->sequence.load(std::memory_order_acquire);
    }

    [[nodiscard]] uint32_t frame_number() const noexcept { return frame_->frame_number; }

private:
    const SeqlockFrame<T>* frame_;
};

// ---------------------------------------------------------------------------
// Boot-time stale segment cleanup  (Gap 4.3)
// ---------------------------------------------------------------------------

/**
 * @brief Remove /dev/shm/nikola_* segments whose mtime predates boot.
 *
 * Call once during Orchestrator::startup() before any shm_open().
 * Safe to call even if /dev/shm does not contain any nikola segments.
 *
 * @return Number of segments removed.
 */
[[nodiscard]] inline int cleanup_stale_shm() noexcept {
    namespace fs = std::filesystem;

    // Determine system boot time by reading the mtime of /proc/1
    // (PID 1 is created at boot and its mtime ≈ boot time).
    std::filesystem::file_time_type boot_time{};
    try {
        boot_time = fs::last_write_time("/proc/1");
    } catch (...) {
        return 0; // Cannot determine boot time — skip cleanup
    }

    int removed = 0;
    const std::string prefix = std::string(SHM_PREFIX.substr(1)); // strip leading /

    try {
        for (const auto& entry : fs::directory_iterator("/dev/shm")) {
            const auto fname = entry.path().filename().string();
            if (!fname.starts_with(prefix)) continue;

            try {
                auto file_time = fs::last_write_time(entry);
                if (file_time < boot_time) {
                    fs::remove(entry);
                    ++removed;
                }
            } catch (...) {
                // Ignore individual file errors
            }
        }
    } catch (...) {
        // /dev/shm not accessible or empty
    }

    return removed;
}

// ---------------------------------------------------------------------------
// Convenience: make a seqlock-backed shared frame mapped to a named segment
// ---------------------------------------------------------------------------

/**
 * @brief Creates a WaveformSHM large enough to hold SeqlockFrame<T>,
 *        returns a pointer to the frame inside the mapping.
 *
 * The caller must keep `shm` alive for as long as the frame pointer is used.
 *
 * @param segment_name  E.g. "/nikola_physics"
 * @param shm           Out: the RAII wrapper (takes ownership)
 * @return              Pointer to SeqlockFrame<T> inside the mapping
 */
template<typename T>
SeqlockFrame<T>* create_seqlock_shm(const std::string& segment_name, WaveformSHM& shm) {
    shm = WaveformSHM(segment_name, SeqlockFrame<T>::byte_size());
    auto* frame = static_cast<SeqlockFrame<T>*>(shm.data());
    // Initialise sequence to 0 (stable state)
    new (&frame->sequence) std::atomic<uint64_t>{0};
    return frame;
}

} // namespace nikola::infrastructure
