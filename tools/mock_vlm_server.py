#!/usr/bin/env python3
"""A fake OpenAI-compatible VLM server.

Not a simulator of any real model -- it is a *test fixture for the harness*. It has
the qualitative shape of a VLM server (vision tokens scale with pixels, prefill
scales with prompt length, everything slows under load, there is a finite batch),
so harness bugs like miscounted TTFT, missing queueing, or a client-bound load
generator show up locally instead of on the 3090 at 2am.

    python3 tools/mock_vlm_server.py --port 8000

Knobs mirror the real ones so an arm YAML can be dry-run unchanged.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PX_PER_VISION_TOKEN = 784  # 28x28, Qwen2.5-VL-style after 2x2 merge

CFG = argparse.Namespace(
    encoder_ms_per_ktok=18.0,   # vision tower, per 1k vision tokens
    prefill_ms_per_ktok=55.0,   # LLM prefill, per 1k prompt tokens
    base_ms=25.0,               # fixed overhead
    itl_ms=12.0,                # decode inter-token latency, unloaded
    max_running=8,              # batch capacity; extra requests queue
    max_pixels=None,            # server-side default cap
)

_inflight = threading.BoundedSemaphore(value=10_000)
_running = 0
_running_lock = threading.Lock()


def _png_dims(raw: bytes) -> tuple[int, int] | None:
    if raw[:8] == b"\x89PNG\r\n\x1a\n" and raw[12:16] == b"IHDR":
        return struct.unpack(">II", raw[16:24])
    return None


def _count_tokens(messages: list, max_pixels: int | None) -> tuple[int, int]:
    """(vision_tokens, text_tokens). Text tokens ~ chars/4, the usual rough rule."""
    vision = 0
    chars = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for part in content or []:
            if part.get("type") == "text":
                chars += len(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if not url.startswith("data:"):
                    vision += 256
                    continue
                raw = base64.b64decode(url.split(",", 1)[1])
                dims = _png_dims(raw)
                px = dims[0] * dims[1] if dims else len(raw) * 3
                if max_pixels:
                    px = min(px, max_pixels)
                vision += max(px // PX_PER_VISION_TOKEN, 1)
    return vision, max(chars // 4, 1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    # ---- chunked SSE helpers ------------------------------------------

    def _begin_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _chunk(self, text: str):
        data = text.encode()
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _end_stream(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- routes --------------------------------------------------------

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            self._json({"object": "list",
                        "data": [{"id": "mock-vlm-7b", "object": "model"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.path.startswith("/v1/chat/completions"):
            self._json({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")

        max_px = CFG.max_pixels
        mmkw = req.get("mm_processor_kwargs") or {}
        if "max_pixels" in mmkw:
            max_px = mmkw["max_pixels"]

        vision, text = _count_tokens(req.get("messages", []), max_px)
        prompt_tokens = vision + text
        max_tokens = int(req.get("max_tokens", 128))

        global _running
        with _running_lock:
            _running += 1
            load = _running
        try:
            # Contention: everything degrades roughly linearly past the batch limit.
            slow = max(1.0, load / CFG.max_running)
            ttft_s = (
                CFG.base_ms
                + CFG.encoder_ms_per_ktok * vision / 1000.0
                + CFG.prefill_ms_per_ktok * prompt_tokens / 1000.0
            ) / 1000.0 * slow
            itl_s = CFG.itl_ms / 1000.0 * slow

            if not req.get("stream"):
                time.sleep(ttft_s + itl_s * max_tokens)
                self._json({
                    "id": "cmpl-mock", "object": "chat.completion",
                    "choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant", "content": "mock"}}],
                    "usage": {"prompt_tokens": prompt_tokens,
                              "completion_tokens": max_tokens,
                              "total_tokens": prompt_tokens + max_tokens},
                })
                return

            self._begin_stream()
            time.sleep(ttft_s)

            def frame(delta=None, usage=None):
                obj = {"id": "cmpl-mock", "object": "chat.completion.chunk",
                       "model": req.get("model", "mock-vlm-7b"),
                       "choices": ([{"index": 0, "delta": delta, "finish_reason": None}]
                                   if delta is not None else [])}
                if usage:
                    obj["usage"] = usage
                return f"data: {json.dumps(obj)}\n\n"

            self._chunk(frame({"role": "assistant"}))  # role-only opener, no content
            for i in range(max_tokens):
                if i:
                    time.sleep(itl_s)
                self._chunk(frame({"content": f"t{i} "}))
            self._chunk(frame(usage={"prompt_tokens": prompt_tokens,
                                     "completion_tokens": max_tokens,
                                     "total_tokens": prompt_tokens + max_tokens}))
            self._chunk("data: [DONE]\n\n")
            self._end_stream()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with _running_lock:
                _running -= 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--itl-ms", type=float, default=CFG.itl_ms)
    ap.add_argument("--max-running", type=int, default=CFG.max_running)
    ap.add_argument("--prefill-ms-per-ktok", type=float, default=CFG.prefill_ms_per_ktok)
    ap.add_argument("--max-pixels", type=int, default=None)
    a = ap.parse_args()
    CFG.itl_ms, CFG.max_running = a.itl_ms, a.max_running
    CFG.prefill_ms_per_ktok, CFG.max_pixels = a.prefill_ms_per_ktok, a.max_pixels

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"mock VLM on http://127.0.0.1:{a.port}  (batch={CFG.max_running}, itl={CFG.itl_ms}ms)",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
