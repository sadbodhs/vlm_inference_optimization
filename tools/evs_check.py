#!/usr/bin/env python3
"""Exploratory feasibility check for EVS (vLLM --video-pruning-rate). NOT an experiment.

Does EVS run on our models in vLLM v0.29.0, and what does it do to prompt tokens on
MEVA windows? For a few gate-fired windows (results/r1b/windows.json), against the
server already running:

  img2   the two VLM frames (offsets 18, 48) as two images -- what every run so far sent
  vid2   the same two frames as one video (data:video/jpeg, 1 fps)
  vid8   eight frames of the window (offsets 6..48, 5 fps) as one video

Run once per server pruning rate; the rate is a server flag, so it is recorded from
--rate, not detected. Writes one JSONL row per request.

    docker/run_harness.sh python3 tools/evs_check.py --rate 0.75 --out results/evs_check/q3vl4b-0.75.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from bench.imaging import resize_to_budget   # noqa: E402
from e7_vlm import QUESTION                  # noqa: E402

MAX_PX = 451_584
VIDEO_Q = QUESTION.replace("These two frames were taken one second apart",
                           "This short clip was taken")


def b64(path: Path) -> str:
    data, _, _, _ = resize_to_budget(path.read_bytes(), MAX_PX)
    return base64.b64encode(data).decode()


def ask(url: str, model: str, content: list, extra: dict) -> dict:
    body = {"model": model, "messages": [{"role": "user", "content": content}],
            "max_tokens": 16, "temperature": 0.0, **extra}
    req = urllib.request.Request(f"{url}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        return {"ok": True, "s": time.perf_counter() - t,
                "prompt_tokens": out["usage"]["prompt_tokens"],
                "text": out["choices"][0]["message"]["content"]}
    except urllib.error.HTTPError as e:
        return {"ok": False, "s": time.perf_counter() - t, "error": e.read().decode()[:400]}
    except Exception as e:                                     # noqa: BLE001
        return {"ok": False, "s": time.perf_counter() - t, "error": repr(e)[:400]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://vlm-server:8000")
    ap.add_argument("--rate", type=float, required=True, help="the server's --video-pruning-rate")
    ap.add_argument("--windows", default="results/r1b/windows.json")
    ap.add_argument("--frames5", default="data/meva/frames5")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--kinds", default="img2,vid2,vid8")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = json.loads(urllib.request.urlopen(f"{args.url}/v1/models").read())["data"][0]["id"]
    wins = json.loads(Path(args.windows).read_text())["windows"]
    pick = wins[:: max(1, len(wins) // args.n)][: args.n]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for w in pick:
            base = w["frames"][0] - 18                      # window start (offsets 18, 48)
            d = Path(args.frames5) / w["clip"]
            two = [b64(d / f"{fr:05d}.jpg") for fr in w["frames"]]
            eight = [b64(d / f"{base + o:05d}.jpg") for o in range(6, 49, 6)]
            reqs = {
                "img2": ([{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}
                          for x in two] + [{"type": "text", "text": QUESTION}], {}),
                "vid2": ([{"type": "video_url", "video_url": {"url": "data:video/jpeg;base64," + ",".join(two)}},
                          {"type": "text", "text": VIDEO_Q}], {"media_io_kwargs": {"video": {"fps": 1}}}),
                "vid8": ([{"type": "video_url", "video_url": {"url": "data:video/jpeg;base64," + ",".join(eight)}},
                          {"type": "text", "text": VIDEO_Q}], {"media_io_kwargs": {"video": {"fps": 5}}}),
            }
            for kind, (content, extra) in reqs.items():
                if kind not in args.kinds.split(","):
                    continue
                r = ask(args.url, model, content, extra)
                row = {"model": model, "rate": args.rate, "window": w["id"], "kind": kind,
                       "truth": w["truth"], **r}
                f.write(json.dumps(row) + "\n")
                print(f"  {kind}  {w['id'][-22:]:22s} "
                      + (f"tok {r['prompt_tokens']:5d}  {r['s']:5.2f}s  {r['text']!r}" if r["ok"]
                         else f"FAILED {r['error'][:200]}"), flush=True)


if __name__ == "__main__":
    main()
