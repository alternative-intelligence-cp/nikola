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

**Spectrum Binning with Octave Folding:**

```cpp
void AudioResonanceEngine::bin_spectrum_to_emitters(
    const std::vector<fftw_complex>& spectrum) {

    // Golden ratio frequencies (Hz)
    const double emitter_freqs[8] = {5.083, 8.225, 13.308, 21.532, 34.840, 56.371, 91.210, 147.58};

    // CRITICAL FIX (MM-AUD-01): Dynamic folding limit instead of hardcoded 200Hz
    // This prevents discarding male voice fundamentals (85-180Hz) and musical notes (D3-G3)
    const double highest_emitter_freq = emitter_freqs[7];  // 147.58 Hz
    const double folding_limit = highest_emitter_freq * 1.5;  // ~221 Hz

    for (int e = 0; e < 8; ++e) {
        double target_freq = emitter_freqs[e];
        double accumulated_magnitude = 0.0;

        // Scan through spectrum and fold octaves
        for (int bin = 0; bin < FFT_SIZE / 2; ++bin) {
            double freq = (double)bin * 44100.0 / FFT_SIZE;

            // Octave folding: map high frequencies down to emitter range
            while (freq > folding_limit) {
                freq *= 0.5;  // Fold down by one octave
            }

            // If this bin folds to our target frequency (within tolerance)
            if (std::abs(freq - target_freq) < 2.0) {  // 2 Hz tolerance
                double magnitude = std::sqrt(spectrum[bin][0] * spectrum[bin][0] +
                                            spectrum[bin][1] * spectrum[bin][1]);
                accumulated_magnitude += magnitude;
            }
        }

        // Set emitter amplitude (accumulated from all octave-folded bins)
        emitters.set_amplitude(e, accumulated_magnitude);
    }
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
    // CRITICAL FIX (Audit 4 Item #7): Make buffer size configurable
    // Problem: Hardcoded 4-frame buffer insufficient for high-latency scenarios or GC pauses
    // Solution: Load from config with safer default (500ms buffer ~= 50 frames at 48kHz/1024)
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
