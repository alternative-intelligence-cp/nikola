#!/usr/bin/env python3
"""
Lightweight Gemini-powered specialist server for Nikola SIE.

Same JSON-Lines protocol as the main Nitpick specialist, but uses Google
Gemini instead of the fine-tuned Mistral model.  This makes first-cycle
testing much faster (no GPU, no 7B model download).

Protocol:
  REQUEST  → {"id": 1, "instruction": "...", "context": "..."}
  RESPONSE ← {"id": 1, "response": "...", "ok": true}
  Ready    ← {"ready": true, "checkpoint": "gemini-2.5-flash"}
"""

import sys
import os
import json

def _log(msg):
    print(f"[specialist-gemini] {msg}", file=sys.stderr, flush=True)

def main():
    # Load API key (first line only — file may have extra text)
    key_path = os.path.expanduser("~/Workspace/CREDS/creds/apiKey.gemini")
    if not os.path.exists(key_path):
        _log(f"API key not found at {key_path}")
        sys.exit(1)

    with open(key_path) as f:
        api_key = f.readline().strip()

    from google import genai
    client = genai.Client(api_key=api_key)

    # Emit ready signal
    ready = json.dumps({"ready": True, "checkpoint": "gemini-2.5-flash"})
    sys.stdout.write(ready + "\n")
    sys.stdout.flush()
    _log("Ready (gemini-2.5-flash)")

    # Serve queries
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            err = json.dumps({"id": None, "ok": False, "error": f"JSON parse error: {e}"})
            sys.stdout.write(err + "\n")
            sys.stdout.flush()
            continue

        req_id = req.get("id")
        instruction = req.get("instruction", "").strip()
        context = req.get("context", "").strip()

        if not instruction:
            resp = json.dumps({"id": req_id, "ok": False, "error": "empty instruction"})
            sys.stdout.write(resp + "\n")
            sys.stdout.flush()
            continue

        prompt = instruction
        if context:
            prompt = f"{instruction}\n\nContext:\n{context}"

        try:
            _log(f"Generating response for request {req_id}...")
            result = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt)
            response_text = result.text
            resp = json.dumps({"id": req_id, "ok": True, "response": response_text})
        except Exception as exc:
            _log(f"Generation error: {exc}")
            resp = json.dumps({"id": req_id, "ok": False, "error": str(exc)})

        sys.stdout.write(resp + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
