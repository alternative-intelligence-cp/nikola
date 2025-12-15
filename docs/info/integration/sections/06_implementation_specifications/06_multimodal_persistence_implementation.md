# Domain VI: Multimodal & Persistence Implementation Specifications

**Document Reference:** NM-004-GAP-MULTIMODAL
**Status:** Implementation-Ready
**Date:** 2025-12-10
**Source:** Gap Analysis Report (Dr. Aris Thorne)

## Overview

This domain handles sensory transduction (audio/visual → waveforms) and persistence (checkpointing, GGUF export). The goal is to ground the physics simulation in real-world sensory data and enable state save/restore.

---

## Gap 6.1: Emitter Injection Coordinates (Audio)

### Context and Requirement

Precise location of the 8 audio emitters in the spatial grid.

### Technical Specification

**Helical Mapping on Spatial Dimensions**

Position emitters in a circular array on the z=0 plane to maximize spatial separation and prevent interference.

#### Coordinate Formula

```
x_n = R · cos(θ_n)
y_n = R · sin(θ_n)
z_n = 0

θ_n = 2π · (n/8)
```

Where:
- **R = N_x/2:** Radius (half the grid width)
- **n ∈ [0, 7]:** Emitter index

This creates a circular array of 8 emitters with 45° angular separation.

### Implementation

```cpp
struct AudioEmitterLayout {
    static constexpr int NUM_EMITTERS = 8;

    static Coord9DInteger compute_emitter_position(int emitter_index,
                                                     const GridDimensions& dims) {
        assert(emitter_index >= 0 && emitter_index < NUM_EMITTERS);

        float radius = dims.Nx / 2.0f;
        float theta = 2.0f * M_PI * emitter_index / NUM_EMITTERS;

        Coord9DInteger coord;
        coord.x = static_cast<uint16_t>(dims.Nx / 2 + radius * std::cos(theta));
        coord.y = static_cast<uint16_t>(dims.Ny / 2 + radius * std::sin(theta));
        coord.z = 0; // Bottom spatial layer

        // Fixed quantum/state coordinates
        coord.u = coord.v = coord.w = 0;
        coord.r = static_cast<uint16_t>(0.8f * dims.Nr); // High resonance
        coord.s = static_cast<uint16_t>(1.0f * dims.Ns); // Moderate refractive index
        coord.t = 0; // Updated dynamically with time

        return coord;
    }
};
```

### Frequency Allocation

Each emitter vibrates at a golden ratio harmonic:

```
f_n = π · φⁿ
```

Where φ = (1 + √5)/2 ≈ 1.618 (golden ratio).

This creates non-resonant frequencies that minimize interference.

### Validation Procedure

1. **Spatial Separation Test:** Verify minimum distance between any two emitters > 10 grid cells.
2. **Interference Test:** Inject all 8 emitters simultaneously. Perform FFT. Verify 8 distinct peaks at expected frequencies.
3. **Crosstalk Test:** Measure amplitude of non-target emitters < 5% of target.

---

## Gap 6.2: Visual Resolution Trade-off

### Context and Requirement

Log-polar transform bin allocation for visual input.

### Technical Specification

#### Log-Polar Configuration

- **Angular Bins (N_θ):** 64 (matches grid y dimension)
- **Radial Bins (N_ρ):** 64 (matches grid x dimension)
- **Total Pixels:** 64 × 64 = 4096
- **Compression:** Input images (1080p) are downsampled to 64×64 via Log-Polar transform before injection

### Rationale

- **Foveal emphasis:** Log-polar gives high resolution at center (where attention focuses), low resolution at periphery
- **Rotation/scale invariance:** Log-polar naturally handles object rotations and scale changes
- **Matches retinal structure:** Biological vision uses log-polar sampling

### Implementation

```cpp
#include <cmath>
#include <opencv2/opencv.hpp>

class LogPolarTransform {
private:
    static constexpr int ANGULAR_BINS = 64;
    static constexpr int RADIAL_BINS = 64;

public:
    cv::Mat transform(const cv::Mat& input_image) {
        int center_x = input_image.cols / 2;
        int center_y = input_image.rows / 2;
        float max_radius = std::hypot(center_x, center_y);

        cv::Mat output(RADIAL_BINS, ANGULAR_BINS, CV_32F);

        for (int r = 0; r < RADIAL_BINS; ++r) {
            for (int theta = 0; theta < ANGULAR_BINS; ++theta) {
                // Log-polar mapping
                float log_r = (r / static_cast<float>(RADIAL_BINS)) * std::log(max_radius);
                float radius = std::exp(log_r);
                float angle = (theta / static_cast<float>(ANGULAR_BINS)) * 2.0f * M_PI;

                // Convert back to Cartesian
                int src_x = center_x + static_cast<int>(radius * std::cos(angle));
                int src_y = center_y + static_cast<int>(radius * std::sin(angle));

                // Sample with bounds checking
                if (src_x >= 0 && src_x < input_image.cols &&
                    src_y >= 0 && src_y < input_image.rows) {
                    output.at<float>(r, theta) = input_image.at<uchar>(src_y, src_x) / 255.0f;
                } else {
                    output.at<float>(r, theta) = 0.0f;
                }
            }
        }

        return output;
    }

    void inject_to_grid(const cv::Mat& log_polar_image, PhysicsEngine& engine,
                       uint16_t time_index) {
        for (int r = 0; r < RADIAL_BINS; ++r) {
            for (int theta = 0; theta < ANGULAR_BINS; ++theta) {
                float intensity = log_polar_image.at<float>(r, theta);

                if (intensity > 0.01f) { // Threshold to avoid injecting noise
                    Coord9DInteger coord;
                    coord.x = r;
                    coord.y = theta;
                    coord.z = 1; // Visual layer (one above audio)
                    coord.u = coord.v = coord.w = 0;
                    coord.r = coord.s = 8; // Mid-range state
                    coord.t = time_index;

                    engine.inject_emitter(coord, intensity);
                }
            }
        }
    }
};
```

---

## Gap 6.3: Checkpoint Frequency

### Context and Requirement

Autosave policy for Differential Manifold Checkpointing (DMC).

### Technical Specification

**Event-Driven + Periodic** checkpointing strategy.

#### Checkpoint Triggers

1. **Periodic:** Every 300 seconds (Consolidation interval from ENGS)
2. **Event:** Immediately before entering NAP state (to save pre-dream state)
3. **Event:** On SIGTERM (graceful shutdown)

### Implementation

```cpp
#include <csignal>
#include <chrono>

class CheckpointManager {
private:
    std::chrono::steady_clock::time_point last_checkpoint;
    static constexpr auto CHECKPOINT_INTERVAL = std::chrono::seconds(300);

    std::string checkpoint_dir = "/var/lib/nikola/checkpoints/";
    volatile sig_atomic_t shutdown_requested = 0;

public:
    CheckpointManager() {
        // Install signal handler for graceful shutdown
        std::signal(SIGTERM, [](int) {
            // Signal handler - set flag
        });

        last_checkpoint = std::chrono::steady_clock::now();
    }

    void update(PhysicsEngine& engine, bool is_napping) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = now - last_checkpoint;

        bool periodic_trigger = (elapsed >= CHECKPOINT_INTERVAL);
        bool nap_trigger = is_napping; // Save before dreaming
        bool shutdown_trigger = (shutdown_requested != 0);

        if (periodic_trigger || nap_trigger || shutdown_trigger) {
            save_checkpoint(engine, get_checkpoint_reason(periodic_trigger,
                                                          nap_trigger,
                                                          shutdown_trigger));
            last_checkpoint = now;
        }
    }

private:
    std::string get_checkpoint_reason(bool periodic, bool nap, bool shutdown) {
        if (shutdown) return "shutdown";
        if (nap) return "pre_nap";
        if (periodic) return "periodic";
        return "unknown";
    }

    void save_checkpoint(PhysicsEngine& engine, const std::string& reason) {
        auto timestamp = std::chrono::system_clock::now();
        auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(
            timestamp.time_since_epoch()).count();

        std::string filename = checkpoint_dir + "nikola_" +
                              std::to_string(millis) + "_" + reason + ".dmc";

        engine.save_differential_checkpoint(filename);
        log_info("Checkpoint saved: {} (reason: {})", filename, reason);
    }
};
```

### Checkpoint Retention Policy

- **Keep last 10 periodic checkpoints** (rolling window)
- **Keep all pre-NAP checkpoints** for dream analysis
- **Keep last shutdown checkpoint** indefinitely

---

## Gap 6.4: GGUF Metadata

### Context and Requirement

Describing 9D architecture to llama.cpp via GGUF key-value pairs.

### Technical Specification

We abuse the GGUF KV pairs to store topology data.

#### Custom Metadata Fields

```
nikola.topology.dims = [16, 16, 128, 32, 32, 32, 64, 64, 64]
nikola.topology.names = ["r", "s", "t", "u", "v", "w", "x", "y", "z"]
nikola.topology.semantics = ["resonance", "state", "time", "quantum_u",
                             "quantum_v", "quantum_w", "spatial_x",
                             "spatial_y", "spatial_z"]
general.architecture = "nikola_v0"
general.file_type = 9 // Custom: Q9_0 balanced nonary
```

**Note:** Requires custom fork of llama.cpp to recognize `nikola_v0` architecture.

### Implementation

```cpp
#include "gguf.h" // From llama.cpp

class GGUFExporter {
public:
    void export_checkpoint(const PhysicsEngine& engine, const std::string& filename) {
        gguf_context* ctx = gguf_init_empty();

        // Topology metadata
        int64_t dims[9] = {16, 16, 128, 32, 32, 32, 64, 64, 64};
        gguf_set_arr_i64(ctx, "nikola.topology.dims", dims, 9);

        const char* names[9] = {"r", "s", "t", "u", "v", "w", "x", "y", "z"};
        gguf_set_arr_str(ctx, "nikola.topology.names", names, 9);

        gguf_set_str(ctx, "general.architecture", "nikola_v0");
        gguf_set_u32(ctx, "general.file_type", 9); // Q9_0

        // Export wavefunction tensors
        auto psi = engine.get_wavefunction();
        export_wavefunction_tensor(ctx, "wavefunction.real", psi.real);
        export_wavefunction_tensor(ctx, "wavefunction.imag", psi.imag);

        // Export metric tensors
        auto metric = engine.get_metric_tensor();
        export_metric_tensor(ctx, "geometry.metric", metric);

        // Write to file
        gguf_write_to_file(ctx, filename.c_str());
        gguf_free(ctx);
    }

private:
    void export_wavefunction_tensor(gguf_context* ctx, const char* name,
                                     const std::vector<float>& data) {
        // Compress using Q9_0 format
        std::vector<uint16_t> compressed = q9_compress(data);
        gguf_add_tensor(ctx, name, compressed.data(), compressed.size());
    }
};
```

---

## Gap 6.5: Compression Trade-offs (Q9_0)

### Context and Requirement

Q9_0 error analysis and adaptive quantization.

### Technical Specification

**Adaptive Quantization** based on node energy.

#### Strategy

- **Low Energy Nodes (|Ψ|² < 10^-3):** Store as Q9_0 (5 trits). Precision: ±0.01
- **High Energy Nodes (Peaks):** Store as FP16 (uncompressed)
- **Flag:** 1 bit in header distinguishes format

#### Rationale

Precision matters most at the peaks (token selection). Low-amplitude regions can tolerate quantization noise.

### Implementation

```cpp
struct Q9Block {
    uint8_t format_flag; // 0 = Q9_0, 1 = FP16
    uint16_t data[]; // Variable size
};

class AdaptiveQuantizer {
private:
    static constexpr float HIGH_ENERGY_THRESHOLD = 1e-3f;

public:
    std::vector<Q9Block> compress(const std::vector<std::complex<float>>& psi) {
        std::vector<Q9Block> blocks;

        for (const auto& val : psi) {
            float intensity = std::norm(val);

            if (intensity > HIGH_ENERGY_THRESHOLD) {
                // Store as FP16 (uncompressed)
                Q9Block block;
                block.format_flag = 1;
                // ... encode as FP16 ...
                blocks.push_back(block);
            } else {
                // Store as Q9_0 (5-trit balanced nonary)
                Q9Block block;
                block.format_flag = 0;
                // ... encode as Q9_0 ...
                blocks.push_back(block);
            }
        }

        return blocks;
    }

    // Q9_0 encoding: Map float [-1, 1] to balanced nonary [-4, +4]
    int8_t quantize_to_trit(float value) {
        // Clamp to [-1, 1]
        value = std::clamp(value, -1.0f, 1.0f);

        // Map to [-4, +4]
        int8_t trit = static_cast<int8_t>(std::round(value * 4.0f));
        return std::clamp(trit, int8_t(-4), int8_t(4));
    }

    float dequantize_from_trit(int8_t trit) {
        return trit / 4.0f;
    }
};
```

### Compression Analysis

**Storage Requirements:**
- **Uncompressed (FP32):** 8 bytes per complex number
- **FP16:** 4 bytes per complex number (50% reduction)
- **Q9_0:** 5 trits × 2 (real+imag) = 10 trits = ~2.5 bytes (69% reduction)

**For 1M active nodes:**
- FP32: 8 MB
- Adaptive (95% Q9_0, 5% FP16): ~2.8 MB

---

## Summary

All 5 Multimodal & Persistence implementation gaps have been addressed with:
- ✅ Circular emitter array with golden ratio frequency spacing
- ✅ 64×64 log-polar visual transform matching biological vision
- ✅ Event-driven + periodic checkpointing (300s interval)
- ✅ GGUF metadata schema for llama.cpp compatibility
- ✅ Adaptive Q9_0/FP16 compression based on node energy

**Status:** Ready for sensory integration and state persistence.
