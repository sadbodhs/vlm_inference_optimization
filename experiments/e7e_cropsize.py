#!/usr/bin/env python3
"""What E7's ROI crops actually sent, against the full frame: sizes, tokens, and the
pixels a person gets. Replays e7_vlm.py's crop rule on the saved detections (no
images decoded, no VLM): roi_box -> crop_jpeg's upscale -> smart_resize's cap.

    python3 experiments/e7e_cropsize.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from bench.imaging import smart_resize
from e7_vlm import DET_SIZE, VLM_OFFSETS, WINDOW, load_dets, roi_box

FW, FH = 1920, 1080
PATCH = 28


def upscale(w: int, h: int, short: int = 448, max_up: float = 3.0) -> tuple[int, int]:
    """crop_jpeg's rule: enlarge so the short side is ~448 px, at most 3x."""
    k = min(max_up, max(1.0, short / max(min(w, h), 1)))
    return (round(w * k), round(h * k)) if k > 1.0 else (w, h)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--det-dir", default="data/meva/det")
    ap.add_argument("--max-pixels", type=int, default=451_584)
    ap.add_argument("--out", default="results/e7e/cropsize.json")
    args = ap.parse_args()

    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    fh, fw = smart_resize(FH, FW, max_pixels=args.max_pixels)
    full_tok = (fh // PATCH) * (fw // PATCH)
    full_scale = fw / FW
    rows = []
    for c in sel:
        dets = load_dets(Path(args.det_dir) / f"{c['clip']}.{DET_SIZE}.jsonl.gz")
        for w in range(idx[c["clip"]]["n_frames"] // WINDOW):
            fr = [w * WINDOW + o for o in VLM_OFFSETS]
            a, b = dets.get(fr[0], []), dets.get(fr[1], [])
            box = roi_box(a, b, FW, FH)
            if box is None:
                continue
            cw, ch = box[2] - box[0], box[3] - box[1]
            uw, uh = upscale(cw, ch)
            sh, sw = smart_resize(uh, uw, max_pixels=args.max_pixels)
            scale = sw / cw                                  # sent px per original px
            ppl = [d for d in a + b if d[0] == "person"]
            hts = [d[5] - d[3] for d in ppl]
            rows.append({"bin": c["bin"], "crop_w": cw, "crop_h": ch,
                         "crop_area_frac": cw * ch / (FW * FH),
                         "sent_w": sw, "sent_h": sh, "tokens": (sh // PATCH) * (sw // PATCH),
                         "scale": scale, "people": len(ppl) // 2 if ppl else 0,
                         "person_h_orig": st.median(hts) if hts else None})

    def q(v, p):
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]

    print(f"original frame        {FW} x {FH} = {FW*FH/1e6:.2f} MP")
    print(f"full frame, as sent   {fw} x {fh}  ->  {full_tok} tokens per frame; "
          f"every object shrunk to {100*full_scale:.0f}% of its size")
    print(f"ROI windows: {len(rows)}\n")
    print(f"{'':28s}{'p10':>9s}{'p50':>9s}{'p90':>9s}{'mean':>9s}")
    def line(lbl, key, fmt="{:9.0f}", mul=1.0):
        v = [r[key] * mul for r in rows if r[key] is not None]
        print(f"{lbl:28s}" + "".join(fmt.format(x) for x in (q(v, .1), q(v, .5), q(v, .9), st.mean(v))))
    line("crop width (orig px)", "crop_w")
    line("crop height (orig px)", "crop_h")
    line("crop area (% of frame)", "crop_area_frac", "{:8.1f}%", 100)
    line("sent width (px)", "sent_w")
    line("sent height (px)", "sent_h")
    line("tokens per frame", "tokens")
    line("person scale (sent/orig)", "scale", "{:9.2f}")
    line("person height, orig (px)", "person_h_orig")
    ratio = [r["scale"] / full_scale for r in rows]
    print(f"{'person pixels vs full frame':28s}" + "".join(f"{x:8.2f}x" for x in
          (q(ratio, .1), q(ratio, .5), q(ratio, .9), st.mean(ratio))))
    print("\nby scene (median):")
    for b in ("empty", "sparse", "moderate", "busy"):
        rb = [r for r in rows if r["bin"] == b]
        if rb:
            print(f"  {b:9s} n={len(rb):4d}  crop {100*st.median(r['crop_area_frac'] for r in rb):5.1f}% of frame"
                  f"  tokens {st.median(r['tokens'] for r in rb):5.0f}"
                  f"  person linear size x{st.median(r['scale'] for r in rb)/full_scale:4.2f} vs full frame")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"full": {"sent_w": fw, "sent_h": fh, "tokens": full_tok, "scale": full_scale},
               "rows": rows}, open(args.out, "w"))


if __name__ == "__main__":
    main()
