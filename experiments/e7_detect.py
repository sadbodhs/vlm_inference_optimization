#!/usr/bin/env python3
"""E7 stage 1: detections for every clip, the VLM's frames, and YOLO's own cost.

One decode pass per clip does three things:

  1. YOLOv8s at 5 fps (every 6th frame), person + vehicle classes, at each input
     size requested -- MEVA's people are often 30-80 px tall in 1080p, which is
     4-10 px at 640 after the letterbox, so input size is a variable, not a setting.
  2. Saves the two frames per 2 s window that the VLM will see. They are chosen
     from the 5 fps grid (offsets 18 and 48 of 60: 0.6 s and 1.6 s) so that the
     ROI crop is built from detections on exactly the frames the VLM gets.
  3. Times the detector with CUDA synchronisation, excluding decode, so the cost
     side of the cascade is measured rather than assumed.

Runs in the CV study's triton-bench:v3 image (ultralytics + torch + cv2):

    docker/run_yolo.sh python3 experiments/e7_detect.py --sizes 640,1280
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics as st
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

FPS = 30
STRIDE = 6                  # 5 fps detection grid
WINDOW = 60                 # 2 s windows
VLM_OFFSETS = (18, 48)      # both on the detection grid
# COCO ids: person, bicycle, car, motorcycle, bus, truck
CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF = 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--video-dir", default="data/meva/video")
    ap.add_argument("--weights", default="/models/yolov8s.pt")
    ap.add_argument("--sizes", default="640,1280")
    ap.add_argument("--det-dir", default="data/meva/det")
    ap.add_argument("--frame-dir", default="data/meva/frames")
    ap.add_argument("--out", default="results/e7/detect_cost.json")
    ap.add_argument("--limit-clips", type=int, default=0)
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    sel = json.load(open(args.selection))["clips"]
    if args.limit_clips:
        sel = sel[: args.limit_clips]
    os.makedirs(args.det_dir, exist_ok=True)

    model = YOLO(args.weights)
    dev = 0 if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        raise SystemExit("no CUDA device visible -- detector cost would be meaningless")
    # warm-up at every size: first-call CUDA init and cudnn autotune are not the
    # steady-state cost of a detector that has been running for hours
    blank = torch.zeros((1080, 1920, 3), dtype=torch.uint8).numpy()
    for s in sizes:
        for _ in range(10):
            model.predict(blank, imgsz=s, conf=CONF, classes=list(CLASSES), device=dev,
                          verbose=False)

    timing = {s: [] for s in sizes}
    per_clip = []
    for c in sel:
        clip = c["clip"]
        path = Path(args.video_dir) / f"{clip}.r13.avi"
        outs = {s: gzip.open(Path(args.det_dir) / f"{clip}.{s}.jsonl.gz", "wt") for s in sizes}
        fdir = Path(args.frame_dir) / clip
        fdir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {path}")
        n, n_det, t0 = 0, 0, time.time()
        while True:
            if n % STRIDE:
                if not cap.grab():
                    break
                n += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if n % WINDOW in VLM_OFFSETS:
                cv2.imwrite(str(fdir / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            for s in sizes:
                torch.cuda.synchronize()
                t = time.perf_counter()
                r = model.predict(frame, imgsz=s, conf=CONF, classes=list(CLASSES),
                                  device=dev, verbose=False)[0]
                torch.cuda.synchronize()
                timing[s].append((time.perf_counter() - t) * 1000)
                boxes = r.boxes
                dets = [[CLASSES[int(k)], round(float(p), 3), *[round(float(v), 1) for v in xyxy]]
                        for k, p, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(),
                                              boxes.xyxy.tolist())]
                n_det += len(dets)
                outs[s].write(json.dumps({"f": n, "d": dets}) + "\n")
            n += 1
        cap.release()
        for f in outs.values():
            f.close()
        per_clip.append({"clip": clip, "frames": n, "wall_s": round(time.time() - t0, 1)})
        print(f"  {clip}: {n} frames, {n_det} detections, {time.time()-t0:.0f}s")
        if n < 8700:
            raise SystemExit(f"{clip}: only {n} frames decoded -- truncated file?")

    cost = {}
    for s in sizes:
        v = sorted(timing[s])
        cost[s] = {"n": len(v), "ms_p50": round(st.median(v), 3),
                   "ms_p90": round(v[int(0.9 * (len(v) - 1))], 3),
                   "ms_mean": round(st.mean(v), 3),
                   "note": "YOLOv8s PyTorch fp32, batch 1, predict() incl. pre/post-processing; "
                           "decode excluded"}
    meta = {"weights": args.weights, "sizes": sizes, "stride": STRIDE, "window": WINDOW,
            "vlm_offsets": VLM_OFFSETS, "classes": CLASSES, "conf": CONF,
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": meta, "cost_ms": cost, "clips": per_clip}, open(args.out, "w"), indent=2)
    for s in sizes:
        print(f"  imgsz {s}: {cost[s]['ms_p50']:.2f} ms/frame p50 ({cost[s]['ms_p90']:.2f} p90)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
