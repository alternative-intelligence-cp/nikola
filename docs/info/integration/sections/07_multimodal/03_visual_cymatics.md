# VISUAL CYMATICS ENGINE

## 24.2 Visual Cymatics Engine

**Status:** MANDATORY - Required for image processing

**Concept:** Map 2D images directly to the toroidal substrate as interference patterns.

## 24.2.1 Mapping Strategy

**Image-to-Torus Mapping:**

| Image Property | Toroidal Mapping | Physics Implementation |
|---------------|------------------|----------------------|
| Pixel (x, y) | Spatial coords $(x, y)$ | Direct lattice addressing |
| Red channel | Emitter 7 amplitude | Modulates $e_7$ ($x$-spatial frequency) |
| Green channel | Emitter 8 amplitude | Modulates $e_8$ ($y$-spatial frequency) |
| Blue channel | Emitter 9 amplitude | Modulates synchronizer |

## 24.2.2 Holographic Property

The image becomes a **standing wave pattern**. Edge detection, blurring, and other convolutions happen naturally via wave propagation rather than explicit kernels.

**Natural Image Operations:**

```
Edge Detection → Wave gradient discontinuities
Blur → Wave diffusion over time
Sharpening → Resonance amplification
Feature Extraction → Harmonic decomposition
```

## 24.2.3 Recognition Mechanism

**Object Recognition Pipeline:**

```
1. Camera captures image
2. Image converted to wave interference pattern
3. Pattern injected into torus
4. System measures resonance with stored patterns
5. IF resonance > threshold:
       Object recognized
```

## 24.2.4 Implementation

**Header Declaration:**

```cpp
// File: include/nikola/multimodal/visual_cymatics.hpp
#pragma once

#include "nikola/physics/torus_manifold.hpp"
#include <opencv2/opencv.hpp>

namespace nikola::multimodal {

class VisualCymaticsEngine {
    TorusManifold& torus;
    EmitterArray& emitters;

public:
    VisualCymaticsEngine(TorusManifold& t, EmitterArray& e);

    void inject_image(const cv::Mat& image);

    double measure_resonance_with_stored_pattern(const std::string& label);

    std::string recognize_object(const cv::Mat& image);

private:
    void map_pixel_to_emitter(int x, int y, const cv::Vec3b& pixel);
};

} // namespace nikola::multimodal
```

## 24.2.5 Core Function

**Image Injection (Holographic Encoding):**

```cpp
void VisualCymaticsEngine::inject_image(const cv::Mat& image) {
    // Resize to torus spatial grid (e.g., 81x81)
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(81, 81));

    // Phase offsets for holographic encoding (in radians)
    const double RED_PHASE_OFFSET = 0.0;           // 0° for red channel
    const double GREEN_PHASE_OFFSET = M_PI / 3.0;  // 60° for green channel
    const double BLUE_PHASE_OFFSET = 2.0 * M_PI / 3.0;  // 120° for blue channel

    // Local wave injection: Each pixel directly modifies the wave state at its spatial coordinate
    // rather than modulating global emitter state. This preserves spatial information across
    // the entire image during holographic encoding.

    for (int y = 0; y < resized.rows; ++y) {
        for (int x = 0; x < resized.cols; ++x) {
            cv::Vec3b pixel = resized.at<cv::Vec3b>(y, x);

            // Map RGB to normalized amplitudes [0, 1]
            double red_amp = pixel[2] / 255.0;
            double green_amp = pixel[1] / 255.0;
            double blue_amp = pixel[0] / 255.0;

            // Spatial coordinate in torus (x, y in dimensions 7, 8)
            Coord9D coord;
            coord.coords = {0, 0, 0, 0, 0, 0, static_cast<int32_t>(x), static_cast<int32_t>(y), 0};

            // Compute local wave interference from three phase-offset channels
            // Each color creates a wave with specific phase and amplitude
            std::complex<double> red_wave(red_amp * cos(RED_PHASE_OFFSET), red_amp * sin(RED_PHASE_OFFSET));
            std::complex<double> green_wave(green_amp * cos(GREEN_PHASE_OFFSET), green_amp * sin(GREEN_PHASE_OFFSET));
            std::complex<double> blue_wave(blue_amp * cos(BLUE_PHASE_OFFSET), blue_amp * sin(BLUE_PHASE_OFFSET));

            // Superposition: add all three color channels
            std::complex<double> combined_wave = red_wave + green_wave + blue_wave;

            // Inject the combined wave LOCALLY at this coordinate
            // This directly modifies the node.wavefunction at (x, y) without affecting global state
            torus.inject_wave_at_coord(coord, combined_wave);
        }
    }

    // Propagate waves for holographic encoding
    // The three phase-offset channels will interfere to create a holographic pattern
    // Natural wave diffusion spreads spatial information across neighboring nodes
    for (int step = 0; step < 100; ++step) {
        torus.propagate(0.01);
    }
}
```

## 24.2.6 Hierarchical Visual Injection

**Multi-Scale Image Pyramid Processing:**

Hierarchical visual injection processes images at multiple resolution levels simultaneously, injecting each scale into distinct frequency bands of the toroidal substrate. This architecture enables scale-invariant object recognition and captures both fine-grained details and coarse structural features.

### 24.2.6.1 Image Pyramid Construction

**Gaussian Pyramid with Frequency Band Mapping:**

```cpp
// File: include/nikola/multimodal/hierarchical_vision.hpp
#pragma once

#include "nikola/multimodal/visual_cymatics.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

namespace nikola::multimodal {

struct PyramidLevel {
    cv::Mat image;
    int level;              // 0 = full resolution, N = coarsest
    double frequency_band;  // Spatial frequency for this scale
    double injection_weight; // Contribution weight to final pattern
};

class HierarchicalVisionEngine {
    TorusManifold& torus;
    VisualCymaticsEngine& base_engine;

    // Pyramid configuration
    static constexpr int NUM_PYRAMID_LEVELS = 5;
    static constexpr double SCALE_FACTOR = 0.5;  // Each level is 50% of previous

    // Frequency band mapping (in radians/pixel)
    // Higher frequencies for fine details, lower for coarse structure
    static constexpr std::array<double, NUM_PYRAMID_LEVELS> FREQUENCY_BANDS = {
        8.0,   // Level 0: Full resolution (81x81) → High frequency
        4.0,   // Level 1: Half resolution (40x40) → Medium-high
        2.0,   // Level 2: Quarter resolution (20x20) → Medium
        1.0,   // Level 3: Eighth resolution (10x10) → Medium-low
        0.5    // Level 4: Sixteenth resolution (5x5) → Low frequency
    };

    // Injection weights (sum to 1.0)
    static constexpr std::array<double, NUM_PYRAMID_LEVELS> LEVEL_WEIGHTS = {
        0.40,  // High-res details: 40%
        0.25,  // Medium-high: 25%
        0.20,  // Medium: 20%
        0.10,  // Medium-low: 10%
        0.05   // Coarse structure: 5%
    };

public:
    HierarchicalVisionEngine(TorusManifold& t, VisualCymaticsEngine& ve)
        : torus(t), base_engine(ve) {}

    std::vector<PyramidLevel> build_pyramid(const cv::Mat& input_image);

    void inject_hierarchical(const cv::Mat& image);

    std::string recognize_multiscale(const cv::Mat& image);

private:
    void inject_pyramid_level(const PyramidLevel& level);
};

} // namespace nikola::multimodal
```

### 24.2.6.2 Pyramid Construction Implementation

**Gaussian Downsampling for Anti-Aliasing:**

```cpp
// File: src/multimodal/hierarchical_vision.cpp

std::vector<PyramidLevel> HierarchicalVisionEngine::build_pyramid(
    const cv::Mat& input_image
) {
    std::vector<PyramidLevel> pyramid;
    pyramid.reserve(NUM_PYRAMID_LEVELS);

    cv::Mat current_level = input_image.clone();

    for (int level = 0; level < NUM_PYRAMID_LEVELS; ++level) {
        // Compute target size for this level
        int target_width = static_cast<int>(81 * std::pow(SCALE_FACTOR, level));
        int target_height = static_cast<int>(81 * std::pow(SCALE_FACTOR, level));

        // Ensure minimum size of 5x5
        target_width = std::max(target_width, 5);
        target_height = std::max(target_height, 5);

        // Apply Gaussian blur before downsampling (anti-aliasing)
        cv::Mat blurred;
        double sigma = 0.5 + (level * 0.3);  // Increasing blur for coarser levels
        cv::GaussianBlur(current_level, blurred, cv::Size(5, 5), sigma);

        // Resize to target resolution
        cv::Mat resized;
        cv::resize(blurred, resized, cv::Size(target_width, target_height),
                   0, 0, cv::INTER_AREA);

        // Create pyramid level
        PyramidLevel pyr_level{
            .image = resized,
            .level = level,
            .frequency_band = FREQUENCY_BANDS[level],
            .injection_weight = LEVEL_WEIGHTS[level]
        };

        pyramid.push_back(pyr_level);

        // Prepare for next iteration
        current_level = resized;
    }

    return pyramid;
}
```

### 24.2.6.3 Multi-Scale Wave Injection

**Frequency-Banded Injection Strategy:**

Each pyramid level is injected into a different spatial frequency band of the torus. This creates a rich, multi-resolution representation where:

- **High-frequency bands** (level 0-1): Capture edges, textures, fine details
- **Medium-frequency bands** (level 2-3): Capture shapes, contours, medium-scale patterns
- **Low-frequency bands** (level 4): Capture overall structure, gross morphology

```cpp
void HierarchicalVisionEngine::inject_pyramid_level(const PyramidLevel& level) {
    const cv::Mat& img = level.image;
    const double freq_band = level.frequency_band;
    const double weight = level.injection_weight;

    // Phase offsets remain the same for RGB holographic encoding
    const double RED_PHASE_OFFSET = 0.0;
    const double GREEN_PHASE_OFFSET = M_PI / 3.0;
    const double BLUE_PHASE_OFFSET = 2.0 * M_PI / 3.0;

    for (int y = 0; y < img.rows; ++y) {
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

            // Normalize amplitudes
            double red_amp = (pixel[2] / 255.0) * weight;
            double green_amp = (pixel[1] / 255.0) * weight;
            double blue_amp = (pixel[0] / 255.0) * weight;

            // Map to spatial coordinates with frequency modulation
            // Scale position based on pyramid level to spread coarse features
            int scale_factor = 1 << level.level;  // 2^level
            int mapped_x = (x * scale_factor) % 81;
            int mapped_y = (y * scale_factor) % 81;

            Coord9D coord;
            coord.coords = {0, 0, 0, 0, 0, 0,
                           static_cast<int32_t>(mapped_x),
                           static_cast<int32_t>(mapped_y), 0};

            // Create carrier waves modulated by frequency band
            // Higher frequency bands create more oscillations per unit distance
            double phase_mod = freq_band * (x + y * 0.1);  // Spatial phase modulation

            std::complex<double> red_wave(
                red_amp * cos(RED_PHASE_OFFSET + phase_mod),
                red_amp * sin(RED_PHASE_OFFSET + phase_mod)
            );

            std::complex<double> green_wave(
                green_amp * cos(GREEN_PHASE_OFFSET + phase_mod),
                green_amp * sin(GREEN_PHASE_OFFSET + phase_mod)
            );

            std::complex<double> blue_wave(
                blue_amp * cos(BLUE_PHASE_OFFSET + phase_mod),
                blue_amp * sin(BLUE_PHASE_OFFSET + phase_mod)
            );

            std::complex<double> combined_wave = red_wave + green_wave + blue_wave;

            // Inject into torus (additive across pyramid levels)
            torus.inject_wave_at_coord(coord, combined_wave);
        }
    }
}

void HierarchicalVisionEngine::inject_hierarchical(const cv::Mat& image) {
    // Build multi-scale pyramid
    auto pyramid = build_pyramid(image);

    // Inject all levels (coarse to fine order for better wave conditioning)
    for (auto it = pyramid.rbegin(); it != pyramid.rend(); ++it) {
        inject_pyramid_level(*it);
    }

    // Propagate to allow multi-scale interference patterns to stabilize
    // Longer propagation than single-scale to allow cross-frequency interactions
    for (int step = 0; step < 200; ++step) {
        torus.propagate(0.01);
    }
}
```

### 24.2.6.4 Scale-Invariant Recognition

**Multi-Resolution Pattern Matching:**

```cpp
std::string HierarchicalVisionEngine::recognize_multiscale(const cv::Mat& image) {
    // Clear previous state
    torus.reset();

    // Inject hierarchical representation
    inject_hierarchical(image);

    // Measure resonance with stored multi-scale patterns
    std::map<std::string, double> resonance_scores;

    std::vector<std::string> known_objects = {
        "cat", "dog", "car", "tree", "person", "building",
        "chair", "bottle", "laptop", "phone"
    };

    for (const auto& label : known_objects) {
        // Measure resonance across all frequency bands
        double total_resonance = 0.0;

        for (int level = 0; level < NUM_PYRAMID_LEVELS; ++level) {
            double band_resonance = base_engine.measure_resonance_with_stored_pattern(
                label + "_L" + std::to_string(level)
            );

            // Weight by pyramid level importance
            total_resonance += band_resonance * LEVEL_WEIGHTS[level];
        }

        resonance_scores[label] = total_resonance;
    }

    // Find maximum weighted resonance
    auto max_elem = std::max_element(
        resonance_scores.begin(),
        resonance_scores.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );

    // Multi-scale recognition has tighter threshold (more discriminative)
    if (max_elem->second > 0.85) {
        return max_elem->first;
    }

    return "unknown";
}
```

### 24.2.6.5 Performance Characteristics

**Computational Complexity:**

- **Pyramid construction:** O(N) where N = total pixels across all levels (≈ 1.33× single-scale)
- **Wave injection:** O(N) across all pyramid levels
- **Propagation steps:** 200 iterations (2× single-scale for cross-frequency stabilization)
- **Recognition:** O(M × L) where M = number of classes, L = pyramid levels

**Memory Footprint:**

- 5 pyramid levels: 81² + 40² + 20² + 10² + 5² = 8,330 pixels total
- Single-scale baseline: 81² = 6,561 pixels
- **Overhead:** 27% additional memory for 5-level pyramid

**Recognition Accuracy Improvements:**

- **Scale invariance:** Recognizes objects at varying distances/sizes
- **Robustness:** Multi-scale voting reduces false positives from single-scale artifacts
- **Feature richness:** Captures both coarse structure and fine texture simultaneously

### 24.2.6.6 Integration with Base Engine

**Unified Vision Pipeline:**

```cpp
// File: include/nikola/multimodal/unified_vision.hpp

class UnifiedVisionPipeline {
    TorusManifold& torus;
    VisualCymaticsEngine base_engine;
    HierarchicalVisionEngine hierarchical_engine;

public:
    UnifiedVisionPipeline(TorusManifold& t, EmitterArray& e)
        : torus(t),
          base_engine(t, e),
          hierarchical_engine(t, base_engine) {}

    // Single-scale fast path (low latency)
    std::string recognize_fast(const cv::Mat& image) {
        return base_engine.recognize_object(image);
    }

    // Multi-scale accurate path (higher accuracy, 2× latency)
    std::string recognize_accurate(const cv::Mat& image) {
        return hierarchical_engine.recognize_multiscale(image);
    }

    // Adaptive: Use hierarchical only if single-scale confidence is low
    std::string recognize_adaptive(const cv::Mat& image) {
        auto result = base_engine.recognize_object(image);

        if (result == "unknown") {
            // Fall back to hierarchical for difficult cases
            return hierarchical_engine.recognize_multiscale(image);
        }

        return result;
    }
};
```

### 24.2.6.7 Applications

**Multi-Scale Vision Use Cases:**

1. **Autonomous Navigation**
   - Detect obstacles at varying distances (near: high-res, far: low-res)
   - Road sign recognition regardless of vehicle distance
   - Pedestrian detection with scale invariance

2. **Medical Imaging**
   - Multi-resolution tumor detection (gross morphology + fine texture)
   - Microscopy analysis across zoom levels
   - Pathology slide scanning at multiple magnifications

3. **Satellite/Aerial Imagery**
   - Building detection from varying altitudes
   - Terrain classification using multi-scale texture
   - Change detection across different resolution datasets

4. **Document Understanding**
   - Layout analysis (coarse) + character recognition (fine)
   - Diagram interpretation with multi-scale structural elements
   - Technical drawing processing across detail levels

## 24.2.7 Pattern Recognition

**Resonance Measurement:**

```cpp
std::string VisualCymaticsEngine::recognize_object(const cv::Mat& image) {
    // 1. Inject image as wave pattern
    inject_image(image);

    // 2. Measure resonance with stored patterns
    std::map<std::string, double> resonance_scores;

    std::vector<std::string> known_objects = {
        "cat", "dog", "car", "tree", "person", "building"
    };

    for (const auto& label : known_objects) {
        double resonance = measure_resonance_with_stored_pattern(label);
        resonance_scores[label] = resonance;
    }

    // 3. Find maximum resonance
    auto max_elem = std::max_element(
        resonance_scores.begin(),
        resonance_scores.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    );

    if (max_elem->second > 0.7) {  // Threshold
        return max_elem->first;
    }

    return "unknown";
}
```

## 24.2.8 Image Processing Operations

**Natural Wave-Based Operations:**

### Edge Detection

Edges appear naturally as regions of high wave gradient:

```cpp
double detect_edge_strength(const Coord9D& coord) {
    auto neighbors = torus.get_neighbors(coord);

    double gradient = 0.0;
    for (const auto& neighbor : neighbors) {
        gradient += std::abs(
            torus.get_amplitude(coord) - torus.get_amplitude(neighbor)
        );
    }

    return gradient / neighbors.size();
}
```

### Image Segmentation

Regions of similar color/intensity form resonant domains:

```cpp
std::vector<Region> segment_image() {
    std::vector<Region> regions;

    // Propagate waves to allow similar regions to resonate
    for (int t = 0; t < 1000; ++t) {
        torus.propagate(0.01);
    }

    // Identify resonant domains
    auto clusters = identify_high_resonance_clusters();

    return clusters;
}
```

## 24.2.9 Video Processing

**Frame-by-Frame Processing:**

```cpp
class VideoProcessor {
    VisualCymaticsEngine& engine;
    cv::VideoCapture capture;

public:
    void process_video(const std::string& video_path) {
        capture.open(video_path);

        cv::Mat frame;
        while (capture.read(frame)) {
            auto result = engine.recognize_object(frame);

            std::cout << "Detected: " << result << std::endl;

            // Process at 30 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(33));
        }
    }
};
```

## 24.2.10 Applications

**Use Cases:**

1. **Document Image Ingestion**
   - Scanned documents converted to wave patterns
   - OCR via resonance matching with character patterns
   - Integration with Section 16 ingestion pipeline

2. **Facial Recognition**
   - Face images stored as unique wave signatures
   - New face compared via resonance measurement
   - Authentication/identification

3. **Object Detection**
   - Real-time camera feed processing
   - Multiple object classes recognized simultaneously
   - Autonomous navigation support

4. **Visual Memory**
   - Images permanently encoded as standing waves
   - Perfect recall through resonance retrieval
   - No separate image database needed

## 24.2.11 Feasibility Assessment

**Feasibility Rank:** MEDIUM

**Rationale:**
- OpenCV integration is straightforward
- Pixel-to-coordinate mapping is simple
- Wave propagation already implemented
- Pattern recognition via resonance requires tuning

**Challenges:**
- Image preprocessing (normalization, resizing)
- Optimal propagation time selection
- Resonance threshold calibration
- Computational cost of repeated wave propagation

**Implementation Effort:** ~1-2 weeks

**Dependencies:**
- OpenCV 4.0+
- Pre-trained object pattern database
- Torus propagation engine (Section 4)

---

**Cross-References:**
- See Section 4 for Wave Interference Physics
- See Section 16 for Autonomous Ingestion Pipeline
- See Section 24 for Cymatic Transduction overview
- See Section 11 for Orchestrator integration
- See OpenCV documentation for image processing
