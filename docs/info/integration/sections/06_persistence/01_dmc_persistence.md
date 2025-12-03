# DIFFERENTIAL MANIFOLD CHECKPOINTING (DMC)

## 19.1 The .nik File Format

**Purpose:** Custom binary format for persisting 9D torus state between sessions.

**Design Principles:**
- Log-structured, append-only
- Differential (only changes since last checkpoint)
- Compressed (Nonary Run-Length Encoding)
- Integrity-verified (Merkle tree root hash)

## 19.2 Binary Structure Specification

**File Layout:**

```
┌────────────────────────────────────┐
│  Global Header (64 bytes)          │
├────────────────────────────────────┤
│  Hyper-Page Block 1                │
│    ├─ Page Header (24 bytes)       │
│    └─ Payload (NRLE compressed)    │
├────────────────────────────────────┤
│  Hyper-Page Block 2                │
│    ├─ Page Header                  │
│    └─ Payload                      │
├────────────────────────────────────┤
│  ...                               │
├────────────────────────────────────┤
│  Footer (128 bytes)                │
│    ├─ Merkle Root (32 bytes)       │
│    └─ Metadata                     │
└────────────────────────────────────┘
```

**Global Header:**

```cpp
struct NikHeader {
    uint32_t magic;           // 0x4E494B4F ("NIKO" in ASCII)
    uint16_t version_major;   // 0
    uint16_t version_minor;   // 4
    uint64_t creation_time;   // Unix timestamp
    uint64_t last_snap_time;  // Timestamp of last full snapshot
    uint8_t  dim_encoding;    // 0x09 (nonary)
    uint8_t  cipher_type;     // 0x01 = ChaCha20-Poly1305
    uint8_t  reserved[38];    // Padding to 64 bytes
} __attribute__((packed));
```

**Hyper-Page Header:**

```cpp
struct PageHeader {
    uint64_t page_id;         // Hilbert index of page center
    uint32_t checksum;        // CRC32C
    uint8_t  flags;           // Bitmask: DIRTY, COMPRESSED, ENCRYPTED, DELETED
    uint32_t payload_len;     // Compressed payload length
    uint8_t  reserved[7];     // Padding to 24 bytes
} __attribute__((packed));

// Flag bits
constexpr uint8_t PAGE_DIRTY      = 0x01;
constexpr uint8_t PAGE_COMPRESSED = 0x02;
constexpr uint8_t PAGE_ENCRYPTED  = 0x04;
constexpr uint8_t PAGE_DELETED    = 0x08;
```

## 19.3 Nonary Run-Length Encoding (NRLE)

**Purpose:** Compress sparse toroidal grid (most nodes are vacuum/zero).

**Algorithm:**

```
Input: Sequence of balanced nonary digits [-4..+4]
Output: Compressed byte stream

Encoding:
- Control trit (1 bit): 0 = Run of zeros, 1 = Raw data
- If control = 0:
    - Length (varint): Number of consecutive zeros
- If control = 1:
    - Count (varint): Number of raw values
    - Data: Packed nonary values (4 bits each)
```

**Implementation:**

```cpp
std::vector<uint8_t> nrle_compress(const std::vector<Nit>& input) {
    std::vector<uint8_t> output;

    size_t i = 0;
    while (i < input.size()) {
        // Count zeros
        size_t zero_count = 0;
        while (i + zero_count < input.size() && input[i + zero_count] == Nit::ZERO) {
            zero_count++;
        }

        if (zero_count > 3) {
            // Encode as run of zeros
            output.push_back(0x00);  // Control bit = 0
            write_varint(output, zero_count);
            i += zero_count;
        } else {
            // Count raw data
            size_t data_count = 0;
            while (i + data_count < input.size() &&
                   input[i + data_count] != Nit::ZERO &&
                   data_count < 255) {
                data_count++;
            }

            if (data_count > 0) {
                // Encode as raw data
                output.push_back(0x01);  // Control bit = 1
                write_varint(output, data_count);

                // Pack values (4 bits each)
                for (size_t j = 0; j < data_count; j += 2) {
                    uint8_t byte = 0;

                    // High nibble
                    byte |= (nit_to_nibble(input[i + j]) << 4);

                    // Low nibble
                    if (j + 1 < data_count) {
                        byte |= nit_to_nibble(input[i + j + 1]);
                    }

                    output.push_back(byte);
                }

                i += data_count;
            } else {
                i++;
            }
        }
    }

    return output;
}

uint8_t nit_to_nibble(Nit nit) {
    // Map [-4..+4] to [0..8]
    return static_cast<uint8_t>(static_cast<int>(nit) + 4);
}

void write_varint(std::vector<uint8_t>& output, size_t value) {
    while (value >= 0x80) {
        output.push_back((value & 0x7F) | 0x80);
        value >>= 7;
    }
    output.push_back(value & 0x7F);
}
```

## 19.4 Nap Cycle and Flush Logic

**Nap Triggers:**

1. Dopamine < 0.2 (fatigue)
2. Dirty cache exceeds 10,000 nodes (pressure)
3. Explicit CLI command: `twi-ctl nap`
4. Scheduled: Every 6 hours

**Nap Sequence:**

```cpp
class PersistenceManager {
    std::map<uint64_t, TorusNode> dirty_cache;
    std::ofstream nik_file;
    std::string nik_path = "/var/lib/nikola/state/main.nik";

public:
    void trigger_nap(const TorusManifold& torus) {
        std::cout << "[NAP] Starting..." << std::endl;

        // 1. Pause emitters (freeze time)
        torus.pause_emitters();

        // 2. Collect dirty nodes
        collect_dirty_nodes(torus);

        // 3. Sort by Hilbert index (sequential writes)
        std::vector<uint64_t> sorted_indices;
        for (const auto& [idx, node] : dirty_cache) {
            sorted_indices.push_back(idx);
        }
        std::sort(sorted_indices.begin(), sorted_indices.end());

        // 4. Group into hyper-pages (3^9 nodes per page)
        std::map<uint64_t, std::vector<TorusNode>> pages;
        for (uint64_t idx : sorted_indices) {
            uint64_t page_id = idx / (19683);  // 3^9
            pages[page_id].push_back(dirty_cache[idx]);
        }

        // 5. Serialize and append
        nik_file.open(nik_path, std::ios::binary | std::ios::app);

        for (const auto& [page_id, nodes] : pages) {
            write_hyper_page(page_id, nodes);
        }

        nik_file.close();

        // 6. Update Merkle root
        update_merkle_root();

        // 7. Clear dirty cache
        dirty_cache.clear();

        // 8. Resume emitters
        torus.resume_emitters();

        std::cout << "[NAP] Complete. Saved " << sorted_indices.size() << " nodes." << std::endl;
    }

private:
    void write_hyper_page(uint64_t page_id, const std::vector<TorusNode>& nodes) {
        PageHeader header;
        header.page_id = page_id;
        header.flags = PAGE_COMPRESSED;

        // Serialize nodes
        std::vector<Nit> nonary_sequence;
        for (const auto& node : nodes) {
            nonary_sequence.push_back(node.nonary_value);
            // (Simplified: would also serialize metric tensor, etc.)
        }

        // Compress
        auto compressed = nrle_compress(nonary_sequence);
        header.payload_len = compressed.size();

        // Checksum
        header.checksum = crc32c(compressed.data(), compressed.size());

        // Write
        nik_file.write(reinterpret_cast<const char*>(&header), sizeof(header));
        nik_file.write(reinterpret_cast<const char*>(compressed.data()), compressed.size());
    }

    uint32_t crc32c(const uint8_t* data, size_t len);
    void collect_dirty_nodes(const TorusManifold& torus);
    void update_merkle_root();
};
```

### 19.4.1 Nap Consolidation Algorithm

**[ADDENDUM]**

The "Nap" is a critical maintenance cycle. It is not merely a pause but a **Memory Consolidation Event**.

**Trigger:** Dopamine < 0.2 OR Boredom > Threshold OR User Command.

**Process:**

1. **Input Gating:** External sensory inputs (CLI, HTTP) are blocked.
2. **Replay (Sharp Wave Ripples):** The system scans the Torus for nodes with high Resonance ($r > 0.9$) but low stability (recently modified).
3. **Transfer:** These patterns are re-injected into the "Long Term" storage sectors (lower frequency resonance bands) with a boosted learning rate ($\eta \times 10$).
4. **Pruning (Neuro-necrosis):** Nodes with Amplitude $< 0.1$ and Resonance $< 0.2$ are de-allocated, freeing up the SHVO hash map.
5. **Snapshot:** A .nik checkpoint is written to disk.

## 19.5 Merkle Tree Integrity

**Purpose:** Verify state hasn't been tampered with.

**Merkle Root Calculation:**

```cpp
std::array<uint8_t, 32> compute_merkle_root(const std::vector<PageHeader>& pages) {
    std::vector<std::array<uint8_t, 32>> hashes;

    // Hash each page
    for (const auto& page : pages) {
        hashes.push_back(sha256_hash(&page, sizeof(page)));
    }

    // Build tree
    while (hashes.size() > 1) {
        std::vector<std::array<uint8_t, 32>> next_level;

        for (size_t i = 0; i < hashes.size(); i += 2) {
            if (i + 1 < hashes.size()) {
                // Combine two hashes
                std::array<uint8_t, 64> combined;
                memcpy(combined.data(), hashes[i].data(), 32);
                memcpy(combined.data() + 32, hashes[i+1].data(), 32);

                next_level.push_back(sha256_hash(combined.data(), 64));
            } else {
                next_level.push_back(hashes[i]);
            }
        }

        hashes = next_level;
    }

    return hashes[0];  // Root
}
```

## 19.6 Implementation

**Complete Persistence System:**

```cpp
class NikolaPersistence {
    PersistenceManager manager;
    std::thread nap_thread;

public:
    void start_auto_nap(TorusManifold& torus, NeurochemistryManager& neuro) {
        nap_thread = std::thread([&]() {
            while (true) {
                // Sleep for 6 hours
                std::this_thread::sleep_for(std::chrono::hours(6));

                // Check dopamine (trigger if fatigued)
                if (neuro.dopamine.get_level() < 0.2) {
                    manager.trigger_nap(torus);
                    neuro.reward(0.05);  // Small reward for nap
                }
            }
        });
    }

    void stop() {
        if (nap_thread.joinable()) {
            nap_thread.join();
        }
    }
};
```

## 19.7 LSM-DMC: Continuous State Streaming

**Status:** MANDATORY - Required for zero data loss

**Current Limitation:** Base DMC only flushes during Nap cycles.

**Enhancement:** Implement a Log-Structured Merge (LSM) tree for continuous streaming writes.

**Architecture:**

```
┌────────────────────────────────────┐
│  Active Nodes (In-Memory)          │
└─────────────┬──────────────────────┘
              ↓ (Dirty writes)
         ┌────┴────┐
         │ MemTable│ (100MB, sorted by Hilbert index)
         └────┬────┘
              ↓ (Flush when full)
         ┌────┴────┐
         │ Level 0 │ (SSTable files)
         └────┬────┘
              ↓ (Compaction)
         ┌────┴────┐
         │ Level 1 │
         └────┬────┘
              ↓
         ┌────┴────┐
         │ Level N │ (.nik files)
         └─────────┘
```

**Benefits:**

- Continuous checkpointing (no data loss on crash)
- Fast writes (sequential log)
- Background compaction (minimal latency impact)

**Implementation Stub:**

```cpp
// File: include/nikola/persistence/lsm_dmc.hpp
#pragma once

#include "nikola/persistence/dmc.hpp"
#include <map>

namespace nikola::persistence {

class LSM_DMC : public PersistenceManager {
    std::map<uint64_t, TorusNode> memtable;  // In-memory sorted table
    const size_t MEMTABLE_SIZE_LIMIT = 100 * 1024 * 1024;  // 100MB

public:
    void write_node(uint64_t hilbert_idx, const TorusNode& node) override;

    void flush_memtable_to_sstable();

    void background_compaction();
};

} // namespace nikola::persistence
```

**Feasibility Rank:** MEDIUM-HIGH (well-understood LSM architecture)

---

**Cross-References:**
- See Section 14 for Neurochemistry triggers
- See Section 22 for Nap System integration
- See Section 20 for GGUF export format
- See Section 5 for Hilbert curve space-filling
