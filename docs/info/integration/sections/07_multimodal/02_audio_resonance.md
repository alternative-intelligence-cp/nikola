# AUDIO RESONANCE ENGINE

## 24.1 Audio Resonance Engine

**Status:** MANDATORY - Core multimodal capability

**Concept:** Map audio frequency spectrum directly to the 8 emitter frequencies.

## 24.1.1 Algorithm

**Processing Pipeline:**

```
1. Audio input (PCM samples)
2. FFT → Frequency spectrum
3. Bin spectrum into 8 channels (corresponding to φ^n emitters)
4. Set emitter amplitudes from bin magnitudes
5. Torus "hears" the sound as physical wave pressure
```

## 24.1.2 Implementation

**Header Declaration:**

```cpp
// File: include/nikola/multimodal/audio_resonance.hpp
#pragma once

#include "nikola/physics/emitter_array.hpp"
#include <fftw3.h>
#include <vector>

namespace nikola::multimodal {

class AudioResonanceEngine {
    EmitterArray& emitters;
    fftw_plan fft_plan;

    const int FFT_SIZE = 4096;
    std::vector<double> input_buffer;
    std::vector<fftw_complex> output_buffer;

public:
    AudioResonanceEngine(EmitterArray& e);
    ~AudioResonanceEngine();

    void process_audio_frame(const std::vector<int16_t>& pcm_samples);

private:
    void bin_spectrum_to_emitters(const std::vector<fftw_complex>& spectrum);
};

} // namespace nikola::multimodal
```

## 24.1.3 Core Processing

**Audio Frame Processing:**

```cpp
void AudioResonanceEngine::process_audio_frame(const std::vector<int16_t>& pcm_samples) {
    // 1. Normalize PCM to [-1.0, 1.0]
    for (size_t i = 0; i < pcm_samples.size() && i < FFT_SIZE; ++i) {
        input_buffer[i] = pcm_samples[i] / 32768.0;
    }

    // 2. Perform FFT
    fftw_execute(fft_plan);

    // 3. Bin spectrum
    bin_spectrum_to_emitters(output_buffer);
}
```

**Spectrum Binning with Anti-Aliased Octave Mapping:**

```cpp
void AudioResonanceEngine::bin_spectrum_to_emitters(
    const std::vector<fftw_complex>& spectrum) {

    // Golden ratio frequencies (Hz)
    const double emitter_freqs[8] = {5.083, 8.225, 13.308, 21.532, 34.840, 56.371, 91.210, 147.58};

    // Nyquist frequency (max frequency in FFT output)
    const double sample_rate = 44100.0;
    const double nyquist_freq = sample_rate / 2.0;  // 22050 Hz
    const double bin_width = sample_rate / FFT_SIZE;

    for (int e = 0; e < 8; ++e) {
        double target_freq = emitter_freqs[e];
        double accumulated_magnitude = 0.0;
        double total_weight = 0.0;

        // Scan through spectrum with anti-aliased octave accumulation
        for (int bin = 0; bin < FFT_SIZE / 2; ++bin) {
            double bin_freq = bin * bin_width;

            // Calculate which octave this bin belongs to relative to target
            // log2(bin_freq / target_freq) gives octave distance
            if (bin_freq < 1.0) continue;  // Skip DC and near-DC bins

            double octave_ratio = bin_freq / target_freq;

            // Check if this bin is harmonically related to target (within 10 octaves)
            if (octave_ratio < 0.5 || octave_ratio > 1024.0) {
                continue;  // Too far from target frequency
            }

            // Calculate octave distance
            double log_ratio = std::log2(octave_ratio);
            double octave_distance = std::abs(log_ratio - std::round(log_ratio));

            // Only accumulate bins that are close to octave multiples (within 5% tolerance)
            if (octave_distance < 0.05) {  // ~3.5% frequency deviation
                int octave = static_cast<int>(std::round(log_ratio));

                // Calculate magnitude
                double magnitude = std::sqrt(spectrum[bin][0] * spectrum[bin][0] +
                                            spectrum[bin][1] * spectrum[bin][1]);

                // Anti-aliasing weight: exponentially decay higher octaves
                // This prevents high-frequency noise from polluting low emitters
                double octave_weight = std::exp(-0.3 * std::abs(octave));  // e^(-0.3|n|)

                // Additional perceptual weighting: A-weighting filter approximation
                // Compensates for human ear sensitivity (boosts 2-5kHz, attenuates low/high)
                double a_weight = calculate_a_weighting(bin_freq);

                double combined_weight = octave_weight * a_weight;

                accumulated_magnitude += magnitude * combined_weight;
                total_weight += combined_weight;
            }
        }

        // Normalize by total weight to prevent loudness variation
        if (total_weight > 1e-6) {
            accumulated_magnitude /= total_weight;
        }

        // Set emitter amplitude with anti-aliased, octave-weighted accumulation
        emitters.set_amplitude(e, accumulated_magnitude);
    }
}

private:
    // A-weighting filter for perceptual audio processing
    // Approximates human ear frequency response (ITU-R 468 weighting)
    double calculate_a_weighting(double freq) {
        // A-weighting transfer function (simplified)
        const double f1 = 20.6;    // Low-frequency pole
        const double f2 = 107.7;   // Mid-frequency pole
        const double f3 = 737.9;   // High-frequency pole
        const double f4 = 12194.0; // Upper-frequency pole

        double f_sq = freq * freq;
        double numerator = f4 * f4 * f_sq * f_sq;
        double denominator = (f_sq + f1 * f1) *
                            std::sqrt((f_sq + f2 * f2) * (f_sq + f3 * f3)) *
                            (f_sq + f4 * f4);

        if (denominator < 1e-10) return 0.0;

        double weight = numerator / denominator;

        // Normalize to [0, 1] range (peak at ~3kHz)
        return std::min(1.0, weight * 0.5);
    }
```

## 24.1.4 Audio Input Sources

**Supported Sources:**

| Source | Format | Sample Rate | Integration |
|--------|--------|-------------|-------------|
| Microphone | PCM 16-bit | 44.1 kHz | ALSA/PulseAudio |
| Audio file | WAV/FLAC | Variable | libsndfile |
| Voice query | Opus codec | 48 kHz | WebRTC |
| Streaming | RTP/UDP | 44.1 kHz | GStreamer |

## 24.1.5 Real-Time Processing

**Latency Requirements:**
- **Target:** < 10ms from audio input to torus injection
- **FFT Size:** 4096 samples (93ms at 44.1kHz)
- **Hop Size:** 2048 samples (50% overlap)
- **Buffer Strategy:** Ring buffer with double buffering

**Lock-Free Ring Buffer Implementation:**

```cpp
// File: include/nikola/types/ring_buffer.hpp
#pragma once

#include <atomic>
#include <vector>
#include <stdexcept>

template<typename T>
class RingBuffer {
    std::vector<T> buffer;
    std::atomic<size_t> write_pos{0};
    std::atomic<size_t> read_pos{0};
    size_t capacity;

public:
    explicit RingBuffer(size_t size)
        : buffer(size + 1),  // One extra slot to distinguish full from empty
          capacity(size + 1) {}

    // Thread-safe write (producer)
    bool write(const T& value) {
        size_t current_write = write_pos.load(std::memory_order_relaxed);
        size_t next_write = (current_write + 1) % capacity;

        // Check if buffer is full
        if (next_write == read_pos.load(std::memory_order_acquire)) {
            return false;  // Buffer full
        }

        buffer[current_write] = value;
        write_pos.store(next_write, std::memory_order_release);
        return true;
    }

    // Thread-safe read (consumer)
    bool read(T& value) {
        size_t current_read = read_pos.load(std::memory_order_relaxed);

        // Check if buffer is empty
        if (current_read == write_pos.load(std::memory_order_acquire)) {
            return false;  // Buffer empty
        }

        value = buffer[current_read];
        read_pos.store((current_read + 1) % capacity, std::memory_order_release);
        return true;
    }

    // Bulk read (for FFT processing)
    std::vector<T> read(size_t count) {
        std::vector<T> result;
        result.reserve(count);

        size_t current_read = read_pos.load(std::memory_order_relaxed);
        size_t current_write = write_pos.load(std::memory_order_acquire);

        // Calculate available samples
        size_t available = (current_write >= current_read)
            ? (current_write - current_read)
            : (capacity - current_read + current_write);

        if (available < count) {
            throw std::runtime_error("Not enough samples in buffer");
        }

        // Read samples
        for (size_t i = 0; i < count; ++i) {
            result.push_back(buffer[current_read]);
            current_read = (current_read + 1) % capacity;
        }

        read_pos.store(current_read, std::memory_order_release);
        return result;
    }

    // Query available samples (thread-safe)
    size_t available() const {
        size_t current_read = read_pos.load(std::memory_order_acquire);
        size_t current_write = write_pos.load(std::memory_order_acquire);

        if (current_write >= current_read) {
            return current_write - current_read;
        } else {
            return capacity - current_read + current_write;
        }
    }

    // Clear buffer
    void clear() {
        read_pos.store(0, std::memory_order_release);
        write_pos.store(0, std::memory_order_release);
    }
};
```

**Performance Optimization:**

```cpp
class RealTimeAudioProcessor {
    std::atomic<bool> running{true};
    // Configurable buffer size for handling high-latency scenarios
    // Default 50 frames (~500ms at 48kHz/1024) handles GC pauses and latency spikes
    size_t buffer_frames;
    RingBuffer<int16_t> audio_buffer;
    std::thread processing_thread;

    RealTimeAudioProcessor() {
        buffer_frames = config.get_int("audio.buffer_frames", 50);  // Default: 50 frames
        audio_buffer = RingBuffer<int16_t>(FFT_SIZE * buffer_frames);
    }

public:
    void start() {
        processing_thread = std::thread([this]() {
            while (running) {
                if (audio_buffer.available() >= FFT_SIZE) {
                    auto samples = audio_buffer.read(FFT_SIZE);
                    engine.process_audio_frame(samples);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        });
    }
};
```

## 24.1.6 Applications

**Use Cases:**

1. **Voice Command Recognition**
   - User speaks command
   - Audio engine extracts frequency profile
   - System matches against stored voice patterns via resonance

2. **Music Analysis**
   - Audio stream contains musical content
   - FFT extracts harmonic structure
   - System recognizes melody/rhythm patterns

3. **Environmental Sound Detection**
   - Background audio monitoring
   - Detect specific sounds (door knock, alarm)
   - Trigger autonomous responses

## 24.1.7 Feasibility Assessment

**Feasibility Rank:** VERY HIGH

**Rationale:**
- FFT is straightforward and well-optimized (FFTW3)
- Frequency binning is simple array mapping
- Real-time audio processing is well-understood
- No complex AI models required

**Implementation Effort:** ~2-3 days

**Dependencies:**
- FFTW3 library
- ALSA/PulseAudio for audio input
- Basic DSP knowledge

---

**Cross-References:**
- See Section 4 for Emitter Array specifications
- See Section 24 for Cymatic Transduction overview
- See Section 11 for Orchestrator integration
- See FFTW3 documentation for FFT optimization
