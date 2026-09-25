#!/usr/bin/env python3
"""R1 (RECIPE.md): the actor-crop reference level.

One sample per MEVA activity with a person actor (positives) and per sampled
tracked person in a window with no labelled activity at all (negatives), each at
one 2 s window's two VLM frames (E7's offsets 18 and 48). Every arm sees the same
samples; only what the VLM is shown changes:

  A0          full frame, E7's prompt
  A1          full frame, red box around the actor box, prompt about the boxed people
  A2-m{1.2,1.5,2,3}   actor box expanded m x, native resolution
  A3-h{112,224,448}   the 2x crop resized so the actor box is h px tall
  A4          the 2x native crop (two frames) + the full frame at <= 112,896 px
  A5          the 2x native crop from tracker boxes matched to the actors

Crops are built from the extracted 1080p JPEGs and passed on losslessly (PNG), so
each image is JPEG-encoded once, by the harness's budget resize -- E7's crops went
through three lossy encodes.

Negatives come from windows with no labelled activity anywhere in the frame, so a
letter is a false alarm under every arm, the full frame included; that is a subset
of RECIPE.md's rule (negatives overlap no annotated actor).

    docker/run_harness.sh python3 experiments/r1_actor_crops.py --arm arms/E_q3vl_4b.yaml
    docker/run_harness.sh python3 experiments/r1_actor_crops.py --arm arms/E_q3vl_8b.yaml \\
        --arms A0,A1,A2-m2,A4
"""
from __future__ import annotations

import asyncio
import base64
import gzip
import io
import json
import random
import statistics as st
from pathlib import Path

from _common import base_parser, resolve_arm
from e7_gate import iou, load_geom, load_types
from e7_vlm import GROUPS, QUESTION, TYPE_TO_GROUP, VLM_OFFSETS, WINDOW

FW, FH = 1920, 1080
MAX_PX = 451_584
CONTEXT_PX = 112_896
PERSON = {"person"}
VEHICLE = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle"}
ALL_ARMS = ["A0", "A1", "A2-m1.2", "A2-m1.5", "A2-m2", "A2-m3",
            "A3-h112", "A3-h224", "A3-h448", "A4", "A5"]

OPTIONS = ("\n".join(f"{g}. {d[0].upper()}{d[1:]}" for g, (d, _) in GROUPS.items())
           + "\nAnswer with every letter that applies, separated by commas, or N if none apply.")
Q_MARK = ("These two frames were taken one second apart by a fixed security camera. A red "
          "box marks one person or group of people, with any vehicle they are using.\n"
          "What are the people inside the red box doing?\n" + OPTIONS)
Q_CROP = ("These two images are close-up crops, taken one second apart, from a fixed "
          "security camera. They are centred on one person or group of people, with any "
          "vehicle they are using.\nWhat are the people in the centre doing?\n" + OPTIONS)
Q_CTX = ("The first two images are close-up crops, taken one second apart, from a fixed "
         "security camera, centred on one person or group of people, with any vehicle they "
         "are using. The third image is the whole camera view at the same moment, for "
         "context.\nWhat are the people in the close-ups doing?\n" + OPTIONS)


def union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def expand(box, m):
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = max(1.0, box[2] - box[0]) * m, max(1.0, box[3] - box[1]) * m
    w, h = max(w, 32.0), max(h, 32.0)          # a sliver at the frame edge breaks resizing
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x1, y1 = min(max(0, x1), FW - int(w)), min(max(0, y1), FH - int(h))   # shift inside
    x1, y1 = max(0, x1), max(0, y1)
    return x1, y1, min(FW, x1 + int(w)), min(FH, y1 + int(h))


def png(im) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def build_samples(args):
    """Positives and negatives with everything each arm needs; seed 0."""
    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    pos, neg_pool, skipped = [], [], {"no_box": 0}
    for c in sel:
        clip, meta = c["clip"], idx[c["clip"]]
        ann = Path(args.repo) / meta["annotation_dir"]
        geom, types = load_geom(ann / f"{clip}.geom.yml"), load_types(ann / f"{clip}.types.yml")
        trk = {}
        for line in gzip.open(Path(args.track_dir) / f"{clip}.jsonl.gz", "rt"):
            if line.strip():
                r = json.loads(line)
                trk[r["f"]] = [(o[0], o[1], tuple(o[3:7])) for o in r["o"]]
        n_win = meta["n_frames"] // WINDOW
        acts = meta["activities"]
        for a in acts:
            kinds = [types.get(t, "?") for t in a["actors"]]
            if not any(k in PERSON for k in kinds):
                continue
            w = min(n_win - 1, ((a["start"] + a["end"]) // 2) // WINDOW)
            fr = [w * WINDOW + o for o in VLM_OFFSETS]
            boxes = [geom[t][f] for t in a["actors"] for f in fr if f in geom.get(t, {})]
            if not boxes:
                skipped["no_box"] += 1
                continue
            ppl_h = [geom[t][f][3] - geom[t][f][1] for t, k in zip(a["actors"], kinds)
                     if k in PERSON for f in fr if f in geom.get(t, {})]
            np_, nv = sum(k in PERSON for k in kinds), sum(k in VEHICLE for k in kinds)
            # A5: the tracker's boxes for the same actors (every annotated actor matched
            # on at least one of the two frames, IoU >= 0.3, same class)
            tboxes, matched_all = [], True
            for t, k in zip(a["actors"], kinds):
                got = False
                for f in fr:
                    g = geom.get(t, {}).get(f)
                    if g is None:
                        continue
                    grp = PERSON if k in PERSON else VEHICLE
                    best = max(((iou(o[2], g), o[2]) for o in trk.get(f, []) if o[0] in grp),
                               default=(0.0, None))
                    if best[0] >= 0.3:
                        tboxes.append(best[1]); got = True
                if any(f in geom.get(t, {}) for f in fr) and not got:
                    matched_all = False
            pos.append({"id": f"{clip}|a{a['id']}", "clip": clip, "w": w, "frames": fr,
                        "kind": "pos", "group": TYPE_TO_GROUP[a["type"]], "type": a["type"],
                        "cls": ("1 person" if np_ == 1 and nv == 0 else
                                "2+ people" if nv == 0 else "person + vehicle"),
                        "actor_h": st.median(ppl_h) if ppl_h else None,
                        "box": union(boxes),
                        "tbox": union(tboxes) if (tboxes and matched_all) else None})
        busy = set()
        for a in acts:
            busy.update(range(a["start"] // WINDOW, a["end"] // WINDOW + 1))
        for w in range(n_win):
            if w in busy:
                continue
            fr = [w * WINDOW + o for o in VLM_OFFSETS]
            ids = {}
            for f in fr:
                for o in trk.get(f, []):
                    if o[0] in PERSON:
                        ids.setdefault(o[1], []).append(o[2])
            for tid, bs in ids.items():
                neg_pool.append({"id": f"{clip}|n{w}-{tid}", "clip": clip, "w": w, "frames": fr,
                                 "kind": "neg", "group": None, "cls": "negative",
                                 "actor_h": st.median(b[3] - b[1] for b in bs),
                                 "box": union(bs), "tbox": union(bs)})
    rng = random.Random(0)
    by_win = {}
    for n in neg_pool:
        by_win.setdefault((n["clip"], n["w"]), []).append(n)
    one_per_window = [rng.choice(v) for _, v in sorted(by_win.items())]
    rng.shuffle(one_per_window)
    neg = one_per_window[:len(pos)]
    return pos, neg, skipped


def render(arm_name, s, frame_dir):
    """(question, [(image bytes, per-image pixel cap)]) for one sample under one arm,
    or None if the arm does not apply (A5 with the actors untracked)."""
    from PIL import Image, ImageDraw
    raws = [(Path(frame_dir) / s["clip"] / f"{f:05d}.jpg").read_bytes() for f in s["frames"]]
    if arm_name == "A0":
        return QUESTION, [(r, MAX_PX) for r in raws]
    ims = [Image.open(io.BytesIO(r)).convert("RGB") for r in raws]
    if arm_name == "A1":
        out = []
        for im in ims:
            d = ImageDraw.Draw(im)
            d.rectangle(s["box"], outline=(255, 0, 0), width=5)
            out.append((png(im), MAX_PX))
        return Q_MARK, out
    box = s["box"]
    if arm_name == "A5":
        if s["tbox"] is None:
            return None
        box = s["tbox"]
    m = float(arm_name.split("-m")[1]) if arm_name.startswith("A2-m") else 2.0
    crop = expand(box, m)
    crops = [im.crop(crop) for im in ims]
    if arm_name.startswith("A3-h"):
        target = int(arm_name.split("-h")[1])
        k = target / max(1.0, box[3] - box[1])
        crops = [c.resize((max(2, round(c.width * k)), max(2, round(c.height * k))),
                          Image.BICUBIC) for c in crops]
    out = [(png(c), MAX_PX) for c in crops]
    if arm_name == "A4":
        return Q_CTX, out + [(raws[1], CONTEXT_PX)]
    return Q_CROP, out


def to_sample(s, question, images):
    from bench.data import Sample
    from bench.imaging import resize_to_budget
    frames, px = [], 0
    for raw, cap in images:
        data, _, h, w = resize_to_budget(raw, cap)
        frames.append(base64.b64encode(data).decode())
        px += h * w
    gold = [s["group"]] if s["group"] else ["N"]
    return Sample(id=s["id"], question=question, answers=gold, frames_b64=frames,
                  image_mime="jpeg", image_px=px)


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--selection", default="results/e7/selection.json")
    p.add_argument("--index", default="data/meva/index.json")
    p.add_argument("--repo", default="data/meva/meva-data-repo")
    p.add_argument("--track-dir", default="data/meva/tracks")
    p.add_argument("--frame-dir", default="data/meva/frames")
    p.add_argument("--arms", default=",".join(ALL_ARMS))
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="samples per kind, for a smoke run")
    p.add_argument("--r1-out", default="results/r1")
    args = p.parse_args()

    from bench.harness import run_arm
    arm = resolve_arm(args)
    arm.max_tokens = 16
    arm.max_pixels = None                   # every cap is applied client-side here
    pos, neg, skipped = build_samples(args)
    if args.limit:
        pos, neg = pos[:args.limit], neg[:args.limit]
    out = Path(args.r1_out); out.mkdir(parents=True, exist_ok=True)
    meta = {"positives": len(pos), "negatives": len(neg), "skipped": skipped,
            "tracked_positives": sum(s["tbox"] is not None for s in pos)}
    (out / "samples.json").write_text(json.dumps({"meta": meta, "samples": pos + neg}))
    print(f"positives {len(pos)}  negatives {len(neg)}  skipped {skipped}  "
          f"tracker-matched positives {meta['tracked_positives']}", flush=True)

    loop = asyncio.get_running_loop()
    for name in args.arms.split(","):
        def build(name=name):
            ss = []
            for s in pos + neg:
                r = render(name, s, args.frame_dir)
                if r is not None:
                    ss.append(to_sample(s, *r))
            return ss
        samples = await loop.run_in_executor(None, build)
        summary = await run_arm(
            arm, samples, mode="closed", concurrency=args.concurrency, scorer=None,
            results_root=out, run_id=f"{name}-{arm.id}",
            unmeasured=["latency under load (closed loop, not a capacity measurement)",
                        "batch-composition effects on greedy decoding"])
        print(f"  {name:9s} n={len(samples):4d}  prompt tokens mean "
              f"{summary['prompt_tokens']['mean']:.0f}  ok {summary['n_ok']}/{len(samples)}",
              flush=True)


if __name__ == "__main__":
    asyncio.run(main())
