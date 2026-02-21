/**
 * @file security/inter_vm_channel.hpp
 * @brief Gap 7.2 — InterVMChannel
 *
 * Host-mediated communication layer between KVM sandbox VMs.
 *
 * Isolation rules (from spec):
 *   - VMs share NO network bridges
 *   - VMs share NO file systems
 *   - Communication is SOLELY host ↔ VM via virtio-serial
 *   - VM A → VM B requires: A→Host (validate) → B
 *
 * This header implements the host-side routing and policy engine.
 * The actual virtio fd I/O is wrapped behind NIKOLA_ENABLE_KVM guards.
 * All policy logic (whitelist, payload validation, routing decisions)
 * is always compiled and testable without a running hypervisor.
 *
 * Allowed communication pairs (whitelist):
 *   executor_1 → orchestrator
 *   executor_2 → orchestrator
 *   (VMs cannot communicate directly with each other)
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nikola::security {

// ============================================================================
// Constants
// ============================================================================

inline constexpr size_t IVM_MAX_PAYLOAD_BYTES = 1u * 1024u * 1024u; // 1 MB

// ============================================================================
// Gap 7.2 — InterVMChannel
// ============================================================================

enum class VMMessageStatus : uint8_t {
    DELIVERED,
    BLOCKED_UNKNOWN_SENDER,
    BLOCKED_UNKNOWN_RECEIVER,
    BLOCKED_POLICY,
    BLOCKED_PAYLOAD_INVALID,
    BLOCKED_PAYLOAD_TOO_LARGE,
};

inline const char* vm_message_status_str(VMMessageStatus s) {
    switch (s) {
        case VMMessageStatus::DELIVERED:                return "DELIVERED";
        case VMMessageStatus::BLOCKED_UNKNOWN_SENDER:   return "BLOCKED_UNKNOWN_SENDER";
        case VMMessageStatus::BLOCKED_UNKNOWN_RECEIVER: return "BLOCKED_UNKNOWN_RECEIVER";
        case VMMessageStatus::BLOCKED_POLICY:           return "BLOCKED_POLICY";
        case VMMessageStatus::BLOCKED_PAYLOAD_INVALID:  return "BLOCKED_PAYLOAD_INVALID";
        case VMMessageStatus::BLOCKED_PAYLOAD_TOO_LARGE:return "BLOCKED_PAYLOAD_TOO_LARGE";
    }
    return "UNKNOWN";
}

struct VMMessage {
    std::string           from_vm;
    std::string           to_vm;
    std::vector<uint8_t>  payload;
};

struct VMMessageResult {
    VMMessageStatus status{VMMessageStatus::BLOCKED_POLICY};
    std::string     log_entry;
};

/**
 * VM registration record.  fd=-1 means not connected (test/stub mode).
 */
struct VMConnection {
    std::string vm_name;
    int         virtio_fd{-1};  ///< -1 = stub mode
    int         pid{-1};
    bool        registered{false};
};

/**
 * Host-mediated inter-VM routing engine.
 *
 * Usage:
 *   InterVMChannel ch;
 *   ch.register_vm("executor_1");
 *   ch.register_vm("orchestrator");
 *   auto result = ch.route(msg);
 */
class InterVMChannel {
public:
    /**
     * Callback invoked when a message is successfully delivered.
     * Signature: (from_vm, to_vm, payload_bytes)
     */
    using DeliveryCallback = std::function<void(const std::string&,
                                                 const std::string&,
                                                 const std::vector<uint8_t>&)>;

    InterVMChannel() { install_default_policy(); }

    // ── VM registration ──────────────────────────────────────────────────────

    void register_vm(const std::string& name, int virtio_fd = -1, int pid = -1) {
        vms_[name] = VMConnection{name, virtio_fd, pid, true};
    }

    void unregister_vm(const std::string& name) { vms_.erase(name); }

    bool is_registered(const std::string& name) const {
        return vms_.count(name) > 0;
    }

    size_t registered_count() const { return vms_.size(); }

    // ── Policy ───────────────────────────────────────────────────────────────

    /**
     * Add an allowed (from→to) pair.
     * By default only executor_N→orchestrator is allowed.
     */
    void allow(const std::string& from, const std::string& to) {
        policy_.emplace(from, to);
    }

    void deny(const std::string& from, const std::string& to) {
        policy_.erase({from, to});
    }

    bool is_allowed(const std::string& from, const std::string& to) const {
        return policy_.count({from, to}) > 0;
    }

    // ── Routing ──────────────────────────────────────────────────────────────

    /**
     * Route a message through the host according to security policy.
     * Steps: validate sender → validate receiver → policy check →
     *        payload scan → deliver.
     */
    VMMessageResult route(const VMMessage& msg) {
        VMMessageResult result;

        // 1. Validate sender
        if (!vms_.count(msg.from_vm)) {
            result.status    = VMMessageStatus::BLOCKED_UNKNOWN_SENDER;
            result.log_entry = "Unknown sender VM: " + msg.from_vm;
            ++stats_.blocked;
            return result;
        }

        // 2. Validate receiver
        if (!vms_.count(msg.to_vm)) {
            result.status    = VMMessageStatus::BLOCKED_UNKNOWN_RECEIVER;
            result.log_entry = "Unknown receiver VM: " + msg.to_vm;
            ++stats_.blocked;
            return result;
        }

        // 3. Size check (cheap, do before policy)
        if (msg.payload.size() > IVM_MAX_PAYLOAD_BYTES) {
            result.status    = VMMessageStatus::BLOCKED_PAYLOAD_TOO_LARGE;
            result.log_entry = "Payload too large from " + msg.from_vm +
                               " (" + std::to_string(msg.payload.size()) + " bytes)";
            ++stats_.blocked;
            return result;
        }

        // 4. Policy whitelist
        if (!is_allowed(msg.from_vm, msg.to_vm)) {
            result.status    = VMMessageStatus::BLOCKED_POLICY;
            result.log_entry = "Policy blocked: " + msg.from_vm + " → " + msg.to_vm;
            ++stats_.blocked;
            return result;
        }

        // 5. Payload content scan (shellcode signatures, etc.)
        if (!validate_payload(msg.payload)) {
            result.status    = VMMessageStatus::BLOCKED_PAYLOAD_INVALID;
            result.log_entry = "Malicious payload from " + msg.from_vm;
            ++stats_.blocked;
            return result;
        }

        // 6. Deliver
        deliver(msg);
        result.status    = VMMessageStatus::DELIVERED;
        result.log_entry = "Delivered: " + msg.from_vm + " → " + msg.to_vm +
                           " (" + std::to_string(msg.payload.size()) + " bytes)";
        ++stats_.delivered;
        return result;
    }

    // ── Callbacks ────────────────────────────────────────────────────────────

    void set_delivery_callback(DeliveryCallback cb) { on_deliver_ = std::move(cb); }

    // ── Stats ────────────────────────────────────────────────────────────────

    struct Stats {
        uint64_t delivered{0};
        uint64_t blocked{0};
    };
    const Stats& stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

private:
    std::unordered_map<std::string, VMConnection>          vms_;
    std::set<std::pair<std::string, std::string>>          policy_;
    DeliveryCallback                                        on_deliver_;
    Stats                                                   stats_;

    void install_default_policy() {
        policy_.emplace("executor_1", "orchestrator");
        policy_.emplace("executor_2", "orchestrator");
    }

    /**
     * Scan payload for known dangerous byte sequences.
     * - Reject if > IVM_MAX_PAYLOAD_BYTES (already checked above)
     * - Reject common x86_64 shellcode NOP sleds and INT3 padding
     */
    static bool validate_payload(const std::vector<uint8_t>& payload) {
        // Detect NOP sled (>= 16 consecutive 0x90)
        size_t nop_run = 0;
        for (uint8_t b : payload) {
            nop_run = (b == 0x90u) ? nop_run + 1 : 0;
            if (nop_run >= 16) return false;
        }

        // Detect INT3 sled (>= 8 consecutive 0xCC)
        size_t int3_run = 0;
        for (uint8_t b : payload) {
            int3_run = (b == 0xCCu) ? int3_run + 1 : 0;
            if (int3_run >= 8) return false;
        }

        return true;
    }

    void deliver(const VMMessage& msg) {
        // Invoke user callback if set
        if (on_deliver_) {
            on_deliver_(msg.from_vm, msg.to_vm, msg.payload);
        }

#ifdef NIKOLA_ENABLE_KVM
        // Write to virtio-serial fd
        auto it = vms_.find(msg.to_vm);
        if (it != vms_.end() && it->second.virtio_fd >= 0) {
            ::write(it->second.virtio_fd,
                    msg.payload.data(),
                    msg.payload.size());
        }
#endif
    }
};

} // namespace nikola::security
