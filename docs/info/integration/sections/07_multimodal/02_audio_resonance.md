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

**Spectrum Binning:**

```cpp
void AudioResonanceEngine::bin_spectrum_to_emitters(
    const std::vector<fftw_complex>& spectrum) {

    // Golden ratio frequencies (Hz)
    const double emitter_freqs[8] = {5.083, 8.225, 13.308, 21.532, 34.840, 56.371, 91.210, 147.58};

    for (int e = 0; e < 8; ++e) {
        double target_freq = emitter_freqs[e];

        // Find FFT bin closest to target frequency
        int bin = (int)(target_freq * FFT_SIZE / 44100.0);  // Assuming 44.1kHz sample rate

        // Get magnitude
        double magnitude = std::sqrt(spectrum[bin][0] * spectrum[bin][0] +
                                     spectrum[bin][1] * spectrum[bin][1]);

        // Set emitter amplitude
        emitters.set_amplitude(e, magnitude);
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

**Performance Optimization:**

```cpp
class RealTimeAudioProcessor {
    std::atomic<bool> running{true};
    RingBuffer<int16_t> audio_buffer;
    std::thread processing_thread;

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
