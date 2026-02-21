#!/usr/bin/env python3
"""
analyzer_main.py — ACAS Speech Analyzer (Speech-to-Text)

Provides Voice Activity Detection + Automatic Speech Recognition as a
persistent process communicating via stdin/stdout newline-delimited JSON.

Protocol (stdout — one JSON object per line):
  {"type": "status",     "mode": "online|offline", "reason": "...", "version": "1.0.0"}
  {"type": "transcript", "text": "...", "confidence": 0.95, "language": "en", "ts": 1234567890.0}
  {"type": "vad",        "speech": true|false, "ts": 1234567890.0}
  {"type": "error",      "msg": "..."}

stdin:
  Any line is treated as a command (reserved for future use).
  EOF / SIGTERM → clean shutdown.

Dependencies (optional):
  - faster-whisper  (pip install faster-whisper)
  - silero-vad      (pip install silero-vad onnxruntime)
  - sounddevice     (pip install sounddevice)
  - numpy

When dependencies are missing the process runs in OFFLINE mode and emits
the startup status line, then stays alive accepting commands until EOF.
"""

import json
import signal
import sys
import time

VERSION = "1.0.0"
CHUNK_MS = 30     # VAD chunk size (milliseconds)
SAMPLE_RATE = 16000  # Hz, required by both Silero VAD and Whisper

# ---------------------------------------------------------------------------
# Clean-exit on SIGTERM
# ---------------------------------------------------------------------------

_running = True

def _handle_sigterm(signum, frame):   # noqa: ANN001
    global _running
    _running = False

signal.signal(signal.SIGTERM, _handle_sigterm)

# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------

def emit(obj: dict) -> None:
    """Write a single JSON object followed by a newline to stdout."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def emit_status(mode: str, reason: str = "") -> None:
    emit({"type": "status", "mode": mode, "reason": reason, "version": VERSION})


def emit_error(msg: str) -> None:
    emit({"type": "error", "msg": msg})

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_deps() -> tuple[bool, str]:
    """Return (ok, reason) for required runtime dependencies."""
    missing = []
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        missing.append("sounddevice")
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        missing.append("faster-whisper")
    if missing:
        return False, "missing packages: " + ", ".join(missing)
    return True, ""

# ---------------------------------------------------------------------------
# Offline stub loop
# ---------------------------------------------------------------------------

def run_offline(reason: str) -> None:
    """Emit offline status and read stdin until EOF."""
    emit_status("offline", reason)
    try:
        while _running:
            line = sys.stdin.readline()
            if not line:      # EOF
                break
            # Commands (reserved) — silently acknowledge
    except (KeyboardInterrupt, BrokenPipeError):
        pass


# ---------------------------------------------------------------------------
# Online loop (VAD + Whisper ASR)
# ---------------------------------------------------------------------------

def run_online() -> None:
    """Full VAD+ASR pipeline using Silero VAD and faster-whisper."""
    import numpy as np
    import sounddevice as sd
    from faster_whisper import WhisperModel

    # Silero VAD — lazy import (may not be installed as standalone package;
    # it ships embedded in the silero GitHub repo or via the ultralytics fork)
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
        model_vad, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (get_speech_timestamps, _, read_audio, _, _) = vad_utils
        use_vad = True
    except Exception:  # pragma: no cover
        use_vad = False

    # Load Whisper (small/base for CPU efficiency)
    try:
        asr = WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as exc:
        emit_error(f"Whisper model load failed: {exc}")
        return

    emit_status("online")

    chunk_samples = int(SAMPLE_RATE * CHUNK_MS / 1000)
    audio_buffer: list = []
    speech_active = False

    def audio_callback(indata: np.ndarray, frames: int,  # noqa: ANN001
                       time_info, status) -> None:  # noqa: ANN001
        nonlocal speech_active
        audio_buffer.append(indata[:, 0].copy())

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=chunk_samples,
            callback=audio_callback,
        ):
            while _running:
                time.sleep(CHUNK_MS / 1000.0)
                if not audio_buffer:
                    continue

                chunk = audio_buffer.pop(0)

                # VAD gate
                ts = time.time()
                if use_vad:
                    import torch
                    t = torch.tensor(chunk)
                    conf = float(model_vad(t, SAMPLE_RATE).item())
                    is_speech = conf > 0.5
                    emit({"type": "vad", "speech": is_speech, "ts": ts})
                    if not is_speech:
                        speech_active = False
                        continue
                    speech_active = True

                # ASR
                segments, info = asr.transcribe(chunk, language=None)
                for seg in segments:
                    text = seg.text.strip()
                    if text:
                        emit({
                            "type": "transcript",
                            "text": text,
                            "confidence": float(getattr(seg, "no_speech_prob", 0.0)),
                            "language": info.language,
                            "ts": ts,
                        })

    except (KeyboardInterrupt, BrokenPipeError):
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok, reason = _check_deps()
    if ok:
        try:
            run_online()
        except Exception as exc:
            emit_error(str(exc))
            run_offline(f"runtime error: {exc}")
    else:
        run_offline(reason)
