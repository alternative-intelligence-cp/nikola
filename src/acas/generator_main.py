#!/usr/bin/env python3
"""
generator_main.py — ACAS Speech Generator (Text-to-Speech)

Provides streaming Text-to-Speech as a persistent process communicating
via stdin/stdout newline-delimited JSON.

Protocol (stdin — one JSON command per line):
  {"cmd": "speak",  "text": "Hello world", "voice": "en_US-lessac-medium"}
  {"cmd": "stop"}
  {"cmd": "status"}

stdout:
  {"type": "status",   "mode": "online|offline", "reason": "...", "version": "1.0.0"}
  {"type": "speaking", "text": "...", "ts": 1234567890.0}
  {"type": "done",     "text": "...", "duration_s": 1.23, "ts": 1234567890.0}
  {"type": "stopped",  "ts": ...}
  {"type": "error",    "msg": "..."}

Dependencies (optional):
  - piper-tts      (pip install piper-tts)
  - sounddevice    (pip install sounddevice)
  - numpy

When dependencies are missing the process runs in OFFLINE mode: it reads
commands, acknowledges them, but emits "done" events without producing audio
(silent dry-run for pipeline testing).
"""

import json
import signal
import sys
import time

VERSION = "1.0.0"
DEFAULT_VOICE = "en_US-lessac-medium"

# ---------------------------------------------------------------------------
# Clean-exit on SIGTERM
# ---------------------------------------------------------------------------

_running = True

def _handle_sigterm(signum, frame):   # noqa: ANN001
    global _running
    _running = False

signal.signal(signal.SIGTERM, _handle_sigterm)

# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------

def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def emit_status(mode: str, reason: str = "") -> None:
    emit({"type": "status", "mode": mode, "reason": reason, "version": VERSION})


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_deps() -> tuple[bool, str]:
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
        import piper  # noqa: F401
    except ImportError:
        missing.append("piper-tts")
    if missing:
        return False, "missing packages: " + ", ".join(missing)
    return True, ""

# ---------------------------------------------------------------------------
# Command loop helper
# ---------------------------------------------------------------------------

def command_loop(speak_fn):  # noqa: ANN001
    """
    Read stdin line-by-line, parse JSON commands, dispatch to speak_fn.

    speak_fn(text: str, voice: str) should do whatever TTS implementation
    is available — it is called synchronously (blocks until audio completes).
    """
    try:
        for raw_line in sys.stdin:
            if not _running:
                break
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                cmd = json.loads(raw_line)
            except json.JSONDecodeError:
                emit({"type": "error", "msg": f"invalid JSON: {raw_line!r}"})
                continue

            action = cmd.get("cmd", "")
            if action == "speak":
                text  = str(cmd.get("text", ""))
                voice = str(cmd.get("voice", DEFAULT_VOICE))
                if not text:
                    continue
                ts = time.time()
                emit({"type": "speaking", "text": text, "ts": ts})
                t0 = time.time()
                try:
                    speak_fn(text, voice)
                except Exception as exc:
                    emit({"type": "error", "msg": str(exc)})
                emit({"type": "done", "text": text,
                      "duration_s": time.time() - t0, "ts": ts})

            elif action == "stop":
                emit({"type": "stopped", "ts": time.time()})

            elif action == "status":
                # Re-emit current status
                emit({"type": "status", "mode": "online", "version": VERSION})

            # Unknown commands silently ignored

    except (KeyboardInterrupt, BrokenPipeError, EOFError):
        pass

# ---------------------------------------------------------------------------
# Offline stub
# ---------------------------------------------------------------------------

def run_offline(reason: str) -> None:
    emit_status("offline", reason)

    def _stub_speak(text: str, voice: str) -> None:  # noqa: ANN001
        # Simulate a 0.5s synthesis delay (no audio produced)
        time.sleep(0.05)

    command_loop(_stub_speak)

# ---------------------------------------------------------------------------
# Online piper-tts loop
# ---------------------------------------------------------------------------

def run_online() -> None:
    """Full TTS using piper-tts."""
    import io
    import numpy as np
    import sounddevice as sd
    from piper.voice import PiperVoice

    _voice_cache: dict = {}

    def _get_voice(voice_name: str) -> PiperVoice:
        if voice_name not in _voice_cache:
            _voice_cache[voice_name] = PiperVoice.load(voice_name)
        return _voice_cache[voice_name]

    emit_status("online")

    def _piper_speak(text: str, voice: str) -> None:
        pv = _get_voice(voice)
        buf = io.BytesIO()
        with sd.OutputStream(
            samplerate=pv.config.sample_rate,
            channels=1,
            dtype="int16",
        ) as stream:
            for audio_bytes in pv.synthesize_stream_raw(text):
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                stream.write(audio.reshape(-1, 1))

    command_loop(_piper_speak)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ok, reason = _check_deps()
    if ok:
        try:
            run_online()
        except Exception as exc:
            emit({"type": "error", "msg": str(exc)})
            run_offline(f"runtime error: {exc}")
    else:
        run_offline(reason)
