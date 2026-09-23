#!/usr/bin/env python3
"""Pre-extract the 5 fps detection grid (every 6th frame) of each E7 clip as JPEG.

E7b's simulated cameras emit these on a real-time clock instead of decoding H.264
live: a dozen 1080p streams decoded in the measuring process would make the CPU,
not the GPU, the thing being measured. Quality 92 matches the frames the VLM saw
in E7, so both stages see identical pixels.

    docker/run_yolo.sh python3 tools/meva_frames5.py
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

STRIDE = 6


def extract(args):
    clip, video, out = args
    d = Path(out) / clip
    d.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video)
    n = kept = 0
    while True:
        if n % STRIDE:
            if not cap.grab():
                break
        else:
            ok, f = cap.read()
            if not ok:
                break
            p = d / f"{n:05d}.jpg"
            if not p.exists():
                cv2.imwrite(str(p), f, [cv2.IMWRITE_JPEG_QUALITY, 92])
            kept += 1
        n += 1
    return clip, n, kept


def main():
    sel = json.load(open("results/e7/selection.json"))["clips"]
    out = "data/meva/frames5"
    jobs = [(c["clip"], f"data/meva/video/{c['clip']}.r13.avi", out) for c in sel]
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        for clip, n, kept in ex.map(extract, jobs):
            print(f"  {clip}: {n} frames, {kept} kept", flush=True)
            if kept < 1450:
                sys.exit(f"{clip}: only {kept} frames extracted")
    print("done")


if __name__ == "__main__":
    main()
