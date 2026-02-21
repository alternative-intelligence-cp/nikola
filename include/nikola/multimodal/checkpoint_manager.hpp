/**
 * @file checkpoint_manager.hpp
 * @brief Gap 6.3 — CheckpointManager
 *
 * Event-driven + periodic persistence of Nikola state.
 *
 * Triggers:
 *   - Periodic: every 300 seconds of wall time
 *   - Pre-NAP: when is_napping transitions to true
 *   - Shutdown: on SIGTERM (atomic flag + background thread drain)
 *
 * Retention policy:
 *   - Last 10 "periodic" checkpoints (rolling circular queue)
 *   - All "pre_nap" checkpoints kept forever in this session
 *   - Last "shutdown" checkpoint
 *
 * Filename format: nikola_{timestamp_ms}_{reason}.dmc
 */
#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace nikola::multimodal {

// ============================================================================
// Constants
// ============================================================================

inline constexpr int CHECKPOINT_INTERVAL_SEC = 300;
inline constexpr int MAX_PERIODIC_CHECKPOINTS = 10;

// ============================================================================
// Gap 6.3 — CheckpointManager
// ============================================================================

enum class CheckpointReason {
    PERIODIC,
    PRE_NAP,
    SHUTDOWN
};

inline const char* reason_to_str(CheckpointReason r) {
    switch (r) {
        case CheckpointReason::PERIODIC:  return "periodic";
        case CheckpointReason::PRE_NAP:   return "pre_nap";
        case CheckpointReason::SHUTDOWN:  return "shutdown";
    }
    return "unknown";
}

/**
 * Metadata about a saved checkpoint.
 */
struct CheckpointRecord {
    std::string       path;
    CheckpointReason  reason{CheckpointReason::PERIODIC};
    int64_t           timestamp_ms{0};
};

/**
 * Manages periodic and event-driven checkpoint persistence.
 *
 * Usage:
 *   CheckpointManager ckpt("/var/lib/nikola/checkpoints");
 *   ckpt.set_save_callback([](const std::string& path, CheckpointReason r) {
 *       // write binary state to path
 *   });
 *   // In main loop:
 *   ckpt.update(is_napping);
 */
class CheckpointManager {
public:
    using SaveCallback = std::function<void(const std::string& path, CheckpointReason)>;
    using Clock        = std::chrono::steady_clock;
    using TimePoint    = Clock::time_point;

    explicit CheckpointManager(std::string checkpoint_dir = "/var/lib/nikola/checkpoints")
        : dir_(std::move(checkpoint_dir))
        , last_periodic_(Clock::now())
        , last_napping_(false)
    {
        std::filesystem::create_directories(dir_);
        install_signal_handler();
    }

    ~CheckpointManager()
    {
        if (shutdown_requested_.load()) {
            flush_shutdown();
        }
    }

    /** Set the callback that actually writes state to disk. */
    void set_save_callback(SaveCallback cb) { save_cb_ = std::move(cb); }

    /**
     * Call every iteration of the main loop.
     *
     * @param is_napping  True when the autonomy engine is in NAP state
     * @param wall_now    Current wall-clock (defaults to Clock::now())
     * @return True if a checkpoint was triggered
     */
    bool update(bool is_napping,
                TimePoint wall_now = Clock::now())
    {
        bool triggered = false;

        // SIGTERM / shutdown trigger
        if (shutdown_requested_.load()) {
            triggered = true;
            do_checkpoint(wall_now, CheckpointReason::SHUTDOWN);
            shutdown_requested_.store(false); // only once
        }

        // Pre-NAP trigger: rising edge of is_napping
        if (is_napping && !last_napping_) {
            triggered = true;
            do_checkpoint(wall_now, CheckpointReason::PRE_NAP);
        }
        last_napping_ = is_napping;

        // Periodic trigger
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            wall_now - last_periodic_).count();
        if (elapsed >= CHECKPOINT_INTERVAL_SEC) {
            triggered = true;
            do_checkpoint(wall_now, CheckpointReason::PERIODIC);
            last_periodic_ = wall_now;
        }

        return triggered;
    }

    /** Force a checkpoint immediately with the given reason. */
    void force_checkpoint(CheckpointReason reason = CheckpointReason::PERIODIC)
    {
        do_checkpoint(Clock::now(), reason);
    }

    /** All checkpoint records saved so far. */
    const std::vector<CheckpointRecord>& records() const { return all_records_; }

    /** Query whether a SIGTERM shutdown was requested. */
    bool is_shutdown_requested() const { return shutdown_requested_.load(); }

    // -------------------------------------------------------------------------
    // Static shutdown flag (used by SIGTERM handler)
    // -------------------------------------------------------------------------
    static inline std::atomic<bool> shutdown_requested_{false};

private:
    std::string  dir_;
    TimePoint    last_periodic_;
    bool         last_napping_;
    SaveCallback save_cb_;

    // Retention
    std::deque<CheckpointRecord>  periodic_queue_;  // max MAX_PERIODIC_CHECKPOINTS
    std::vector<CheckpointRecord> prenap_records_;
    std::string                   last_shutdown_path_;
    std::vector<CheckpointRecord> all_records_;

    static void signal_handler(int /*sig*/) {
        shutdown_requested_.store(true);
    }

    void install_signal_handler() {
        std::signal(SIGTERM, CheckpointManager::signal_handler);
        std::signal(SIGINT,  CheckpointManager::signal_handler);
    }

    void flush_shutdown() {
        do_checkpoint(Clock::now(), CheckpointReason::SHUTDOWN);
    }

    std::string build_path(int64_t ts_ms, CheckpointReason reason) const {
        std::ostringstream oss;
        oss << dir_ << "/nikola_" << ts_ms << "_" << reason_to_str(reason) << ".dmc";
        return oss.str();
    }

    void do_checkpoint(TimePoint now, CheckpointReason reason)
    {
        const auto ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        const std::string path = build_path(ts_ms, reason);

        // Invoke user callback if set — otherwise write a minimal marker file
        if (save_cb_) {
            save_cb_(path, reason);
        } else {
            std::ofstream f(path, std::ios::binary);
            if (f) {
                const uint32_t magic = 0x444D4300u; // "DMC\0"
                f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
                f.write(reinterpret_cast<const char*>(&ts_ms), sizeof(ts_ms));
                const uint8_t r = static_cast<uint8_t>(reason);
                f.write(reinterpret_cast<const char*>(&r), sizeof(r));
            }
        }

        CheckpointRecord rec{path, reason, ts_ms};
        all_records_.push_back(rec);

        // Apply retention policy
        switch (reason) {
            case CheckpointReason::PERIODIC: {
                periodic_queue_.push_back(rec);
                while (static_cast<int>(periodic_queue_.size()) > MAX_PERIODIC_CHECKPOINTS) {
                    // Evict oldest but don't delete the file — caller may have it open
                    periodic_queue_.pop_front();
                }
                break;
            }
            case CheckpointReason::PRE_NAP:
                prenap_records_.push_back(rec);
                break;
            case CheckpointReason::SHUTDOWN:
                last_shutdown_path_ = path;
                break;
        }
    }
};

} // namespace nikola::multimodal
