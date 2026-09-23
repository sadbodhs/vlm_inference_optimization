#!/usr/bin/env python3
"""E7b: the live cascade -- cameras, detector and VLM on one GPU, in real time.

PLAN.md 11. N simulated cameras replay E7's MEVA clips at 5 fps on a wall clock.
One detector thread (YOLOv8s @ 1280, PyTorch or TensorRT) sees every frame; a gate
decides per 2 s window; fired windows go to the VLM (vLLM, same card) as open-loop
streaming requests. Nothing waits for anything it would not wait for in production:
a camera never waits for the detector, and the detector never waits for the VLM.

Per N: answer staleness (answer complete - capture of the newest frame sent),
detector latency and drops, the gate's live call rate, GPU utilisation and memory.
Supported = >= 95% of answers under 2 s old AND < 1% of detector frames dropped.

    docker/run_yolo.sh python3 experiments/e7_live.py --gate person-motion --arm roi \\
        --detector trt --cams 4,6,8,10,12,14
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import queue
import random
import statistics as st
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import httpx
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from e7_gate import PERSON, VEHICLE, changed  # noqa: E402
from e7_vlm import QUESTION, VLM_OFFSETS, WINDOW, crop_jpeg, roi_box  # noqa: E402
from bench.imaging import resize_to_budget  # noqa: E402

STRIDE = 6
FPS_DET = 5.0
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
STALE_DROP_S = 1.0
FRESH_S = 2.0
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"


def gate_fires(gate: str, win_dets: list[list], prev: list) -> bool:
    if gate == "dense":
        return True
    if gate == "presence":
        return any(d[0] in PERSON | VEHICLE for fd in win_dets for d in fd)
    if gate == "person":
        return any(d[0] in PERSON for fd in win_dets for d in fd)
    groups = PERSON | VEHICLE if gate == "motion" else PERSON
    seq = [prev] + win_dets
    return any(changed(a, b, groups) for a, b in zip(seq, seq[1:]))


class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.util, self.mem, self.stop = [], [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                u, m = out.strip().split(",")
                self.util.append(float(u)); self.mem.append(float(m))
            except Exception:
                pass
            self.stop.wait(1.0)


def _detector_proc(weights, imgsz, in_q, out_q):
    """The detector in its OWN process. In-process, its latency tail could not be
    told apart from the harness: the VLM client's JPEG crop/resize/base64 work holds
    Python's GIL, and a smoke run showed a 23 ms median with a 345 ms p99. Out of
    process, what remains is the GPU -- time-sliced with the VLM -- which is the
    thing E7b exists to measure. Decode runs on a small thread pool here (cv2
    releases the GIL), so the model loop only does detector work."""
    import cv2 as _cv2
    import numpy as _np
    from concurrent.futures import ThreadPoolExecutor as _TPE
    from ultralytics import YOLO
    model = YOLO(weights, task="detect")
    blank = _np.zeros((1080, 1920, 3), _np.uint8)
    for _ in range(10):
        model.predict(blank, imgsz=imgsz, conf=0.25, classes=list(CLASSES), device=0, verbose=False)
    out_q.put(("ready",))
    dec = _TPE(max_workers=4)
    pending = []
    while True:
        # keep up to 8 decodes in flight, preserving arrival order
        while len(pending) < 8:
            try:
                item = in_q.get(timeout=0.005 if pending else None)
            except Exception:
                break
            if item is None:
                pending.append((None, None))
                break
            cam, fno, path, t_cap = item
            fut = dec.submit(lambda p=path: _cv2.imread(p, _cv2.IMREAD_COLOR))
            pending.append(((cam, fno, t_cap), fut))
        if not pending:
            continue
        meta, fut = pending.pop(0)
        if meta is None:
            out_q.put(("done",))
            return
        cam, fno, t_cap = meta
        img = fut.result()
        if time.time() - t_cap > STALE_DROP_S:
            out_q.put(("res", cam, fno, None, t_cap, time.time()))
            continue
        r = model.predict(img, imgsz=imgsz, conf=0.25, classes=list(CLASSES), device=0, verbose=False)[0]
        b = r.boxes
        dets = [[CLASSES[int(k)], float(p), *map(float, xyxy)]
                for k, p, xyxy in zip(b.cls.tolist(), b.conf.tolist(), b.xyxy.tolist())]
        out_q.put(("res", cam, fno, dets, t_cap, time.time()))


class Detector:
    """Parent-side handle: a child process plus a thread relaying its results."""

    def __init__(self, weights: str, imgsz: int, loop, on_result):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.in_q, self.out_q = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_detector_proc, args=(weights, imgsz, self.in_q, self.out_q),
                                daemon=True)
        self.proc.start()
        assert self.out_q.get(timeout=600)[0] == "ready"
        self.loop, self.on_result = loop, on_result
        self.lat, self.dropped, self.done = [], 0, 0
        self.relay = threading.Thread(target=self._relay, daemon=True)
        self.relay.start()

    def _relay(self):
        while True:
            msg = self.out_q.get()
            if msg[0] == "done":
                return
            _, cam, fno, dets, t_cap, t_done = msg
            if dets is None:
                self.dropped += 1
            else:
                self.done += 1
                self.lat.append(t_done - t_cap)
            self.loop.call_soon_threadsafe(self.on_result, cam, fno, dets)

    def put(self, cam, fno, path, t_cap):
        self.in_q.put((cam, fno, str(path), t_cap))

    def close(self):
        self.in_q.put(None)
        self.relay.join(timeout=30)
        self.proc.join(timeout=30)


async def run_once(args, n_cams: int, clips: list[dict]) -> dict:
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=8)
    rng = random.Random(1000 + n_cams)
    frames_root = Path(args.frame_dir)

    cams = []
    for i in range(n_cams):
        c = clips[i % len(clips)]
        n_frames = 9000
        start = rng.randrange(0, n_frames // WINDOW - 10) * WINDOW
        cams.append({"clip": c["clip"], "bin": c["bin"], "start": start, "n": n_frames,
                     "dets": {}, "raw": {}, "tcap": {}})

    results, pending = [], set()
    t0 = time.time() + 2.0                      # common start
    warm_until = t0 + args.warmup
    end_at = warm_until + args.duration
    counters = {"windows": 0, "fired": 0}
    http = httpx.AsyncClient(limits=httpx.Limits(max_connections=512), timeout=120)

    async def send(cam: dict, w_abs: int, fr: list[int], t_newest: float, dets_pair):
        raws = [cam["raw"].get(f) for f in fr]
        if any(r is None for r in raws):
            return
        def build():
            imgs = raws
            if args.arm == "roi":
                box = roi_box(dets_pair[0] or [], dets_pair[1] or [], 1920, 1080)
                if box is not None:
                    imgs = [crop_jpeg(r, box) for r in raws]
            return [base64.b64encode(resize_to_budget(r, args.max_pixels)[0]).decode()
                    for r in imgs]
        b64 = await loop.run_in_executor(pool, build)
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}
                   for x in b64] + [{"type": "text", "text": QUESTION}]
        body = {"model": MODEL_ID, "messages": [{"role": "user", "content": content}],
                "max_tokens": 16, "temperature": 0, "stream": True}
        t_send, t_first, ok = time.time(), None, False
        try:
            async with http.stream("POST", f"{args.base_url}/v1/chat/completions", json=body) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        if t_first is None and json.loads(line[6:])["choices"][0]["delta"].get("content"):
                            t_first = time.time()
                ok = r.status_code == 200
        except Exception:
            ok = False
        t_done = time.time()
        results.append({"cam": cam["clip"], "w": w_abs, "ok": ok,
                        "staleness_s": t_done - t_newest, "ttft_s": (t_first or t_done) - t_send,
                        "queue_s": t_send - t_newest, "measured": t_newest >= warm_until})

    def spawn(coro):
        t = asyncio.ensure_future(coro)
        pending.add(t)
        t.add_done_callback(pending.discard)

    def window_ready(ci: int, w_abs: int):
        cam = cams[ci]
        base = w_abs * WINDOW
        fr = [base + o for o in VLM_OFFSETS]
        win = [cam["dets"].get(f) or [] for f in range(base, base + WINDOW, STRIDE)]
        prev = cam["dets"].get(base - STRIDE) or []
        measured = cam["tcap"].get(base + WINDOW - STRIDE, 0) >= warm_until
        if measured:
            counters["windows"] += 1
        if gate_fires(args.gate, win, prev):
            if measured:
                counters["fired"] += 1
            t_newest = cam["tcap"].get(fr[1])
            if t_newest is not None:
                spawn(send(cam, w_abs, fr, t_newest,
                           (cam["dets"].get(fr[0]), cam["dets"].get(fr[1]))))

    def on_result(ci: int, fno: int, dets):
        cams[ci]["dets"][fno] = dets
        if fno % WINDOW == WINDOW - STRIDE:           # last sampled frame of its window
            window_ready(ci, fno // WINDOW)

    det = None
    if args.gate != "dense":
        weights = "/models/yolov8s.pt" if args.detector == "pytorch" else "/models/e7/yolov8s.engine"
        det = Detector(weights, 1280, loop, on_result)

    sampler = GpuSampler(); sampler.start()

    lateness = []     # R2: is the client keeping up? A late frame is the harness, not the GPU

    async def camera(ci: int):
        cam = cams[ci]
        k = 0
        while True:
            t = t0 + k / FPS_DET
            if t > end_at:
                return
            await asyncio.sleep(max(0.0, t - time.time()))
            if t >= warm_until:
                lateness.append(time.time() - t)
            fno = cam["start"] + k * STRIDE
            if fno >= cam["n"]:
                cam["start"] -= cam["n"]; fno -= cam["n"]      # loop the clip
                cam["dets"].clear(); cam["raw"].clear(); cam["tcap"].clear()
            path = frames_root / cam["clip"] / f"{fno % cam['n']:05d}.jpg"
            raw = await loop.run_in_executor(pool, path.read_bytes)
            cam["tcap"][fno] = t
            if fno % WINDOW in VLM_OFFSETS:
                cam["raw"][fno] = raw
            if det is not None:
                det.put(ci, fno, path, t)
            elif fno % WINDOW == VLM_OFFSETS[1]:       # dense, no detector: send at once
                w_abs = fno // WINDOW
                if t >= warm_until:
                    counters["windows"] += 1; counters["fired"] += 1
                spawn(send(cam, w_abs, [w_abs * WINDOW + o for o in VLM_OFFSETS], t, (None, None)))
            # forget frames two windows back
            for d in (cam["dets"], cam["raw"], cam["tcap"]):
                for old in [f for f in d if f < fno - 2 * WINDOW]:
                    d.pop(old, None)
            k += 1

    await asyncio.gather(*(camera(i) for i in range(n_cams)))
    # drain: in-flight answers still count; anything unfinished after 30 s is stale anyway
    if pending:
        await asyncio.wait(pending, timeout=30)
    if det is not None:
        det.close()
    sampler.stop.set()
    await http.aclose()
    pool.shutdown(wait=False)

    m = [r for r in results if r["measured"]]
    ok = [r for r in m if r["ok"]]
    stale = sorted(r["staleness_s"] for r in ok)
    fresh = sum(1 for s in stale if s <= FRESH_S) / len(m) if m else None
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None
    det_total = (det.done + det.dropped) if det else 0
    drop = det.dropped / det_total if det_total else 0.0
    row = {
        "cams": n_cams, "gate": args.gate, "arm": args.arm,
        "detector": args.detector if det else "none",
        "windows": counters["windows"], "sent": len(m), "ok": len(ok),
        "call_rate": counters["fired"] / counters["windows"] if counters["windows"] else None,
        "staleness_p50_s": q(stale, 0.5), "staleness_p95_s": q(stale, 0.95),
        "fresh_fraction": fresh,
        "ttft_p50_s": q(sorted(r["ttft_s"] for r in ok), 0.5),
        "det_frames": det_total, "det_drop_frac": drop,
        "det_latency_p50_s": q(sorted(det.lat), 0.5) if det else None,
        "det_latency_p99_s": q(sorted(det.lat), 0.99) if det else None,
        "emit_late_p99_s": q(sorted(lateness), 0.99),
        "client_bound": bool(lateness and q(sorted(lateness), 0.99) > 0.1),
        "gpu_util_mean": st.mean(sampler.util) if sampler.util else None,
        "gpu_mem_max_mib": max(sampler.mem) if sampler.mem else None,
        "supported": bool(fresh is not None and fresh >= 0.95 and drop < 0.01),
    }
    return row, results


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="person-motion",
                    choices=["dense", "presence", "person", "motion", "person-motion"])
    ap.add_argument("--arm", default="roi", choices=["full", "roi"])
    ap.add_argument("--detector", default="pytorch", choices=["pytorch", "trt"])
    ap.add_argument("--cams", default="2,4,6,8")
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--max-pixels", type=int, default=451584)
    ap.add_argument("--base-url", default="http://vlm-server:8000")
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--frame-dir", default="data/meva/frames5")
    ap.add_argument("--out", default="results/e7b")
    ap.add_argument("--stop-after-fails", type=int, default=2)
    args = ap.parse_args()

    sel = json.load(open(args.selection))["clips"]
    bins = ["empty", "sparse", "moderate", "busy"]
    by = {b: [c for c in sel if c["bin"] == b] for b in bins}
    clips = [by[b][i] for i in range(6) for b in bins]      # interleaved scene mix

    tag = f"{args.gate}-{args.arm}-{args.detector if args.gate != 'dense' else 'none'}"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, fails = [], 0
    for n in [int(x) for x in args.cams.split(",")]:
        row, detail = await run_once(args, n, clips)
        rows.append(row)
        (out / f"{tag}-n{n}.jsonl").write_text("\n".join(json.dumps(r) for r in detail) + "\n")
        print(f"  N={n:<3} calls {100*(row['call_rate'] or 0):5.1f}%  sent {row['sent']:4d}  "
              f"fresh {100*(row['fresh_fraction'] or 0):5.1f}%  stale p95 "
              f"{row['staleness_p95_s'] or 0:5.2f}s  det drop {100*row['det_drop_frac']:4.1f}%  "
              f"det p99 {row['det_latency_p99_s'] or 0:5.3f}s  gpu {row['gpu_util_mean'] or 0:4.0f}%  "
              f"mem {row['gpu_mem_max_mib'] or 0:6.0f}  late p99 {row['emit_late_p99_s'] or 0:.3f}s  "
              f"{'CLIENT-BOUND ' if row['client_bound'] else ''}{'OK' if row['supported'] else 'no'}",
              flush=True)
        fails = 0 if row["supported"] else fails + 1
        if fails >= args.stop_after_fails:
            break
        await asyncio.sleep(5)
    sup = [r["cams"] for r in rows if r["supported"]]
    summary = {"meta": {"experiment": "e7b-live", "gate": args.gate, "arm": args.arm,
                        "detector": args.detector, "duration_s": args.duration,
                        "warmup_s": args.warmup, "fresh_s": FRESH_S,
                        "stale_drop_s": STALE_DROP_S, "max_pixels": args.max_pixels},
               "rows": rows, "max_supported": max(sup) if sup else 0}
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    print(f"  {tag}: max supported cameras = {summary['max_supported']}")


if __name__ == "__main__":
    asyncio.run(main())
