#!/usr/bin/env python3
"""E7 stage 3: what the VLM recognises, full frame vs region-of-interest crop.

Each arm is run ONCE over every window it applies to; cascades are then evaluated
by replay (e7_report.py). That is exact, not an approximation: a gate only decides
*whether* a window is sent, never what the VLM sees, so the answer for a sent
window is the same whichever gate sent it.

Arms (PLAN.md 10):
  full   both frames of the window, whole 1080p frame, capped at --max-pixels each
  roi    both frames cropped to the union of the detector's boxes (people if any,
         else everything), padded, same crop on both frames. Windows with no
         detection on either VLM frame have nothing to crop and are skipped -- no
         detection-driven gate can fire on them from those frames anyway.

The VLM answers a fixed multi-select over eight activity groups that partition
MEVA's 36 types. Scoring is per activity *instance*: recognised if any window
overlapping it contains its group's letter.

    docker/run_harness.sh python3 experiments/e7_vlm.py --arm arms/B0_vllm_awq_clean.yaml --arms full,roi
"""
from __future__ import annotations

import asyncio
import gzip
import io
import json
from pathlib import Path

from _common import base_parser, resolve_arm

WINDOW = 60
VLM_OFFSETS = (18, 48)
DET_SIZE = 1280          # ROI boxes come from the higher-recall detector pass

GROUPS = {
    "A": ("a person gets into or out of a vehicle, opens or closes a vehicle door or "
          "trunk, or loads or unloads a vehicle",
          ["person_opens_vehicle_door", "person_closes_vehicle_door", "person_enters_vehicle",
           "person_exits_vehicle", "person_opens_trunk", "person_closes_trunk",
           "person_loads_vehicle", "person_unloads_vehicle", "vehicle_drops_off_person",
           "vehicle_picks_up_person"]),
    "B": ("a vehicle starts, stops, turns, or reverses",
          ["vehicle_starts", "vehicle_stops", "vehicle_turns_left", "vehicle_turns_right",
           "vehicle_reverses", "vehicle_makes_u_turn"]),
    "C": ("a person walks through a doorway or opens or closes a door",
          ["person_enters_scene_through_structure", "person_exits_scene_through_structure",
           "person_opens_facility_door", "person_closes_facility_door"]),
    "D": ("a person picks up, puts down, carries, or hands over an object",
          ["person_picks_up_object", "person_puts_down_object", "person_carries_heavy_object",
           "person_transfers_object", "person_steals_object"]),
    "E": ("a person uses a phone", ["person_texts_on_phone", "person_talks_on_phone"]),
    "F": ("people talk to or touch each other",
          ["person_talks_to_person", "person_embraces_person", "hand_interacts_with_person"]),
    "G": ("a person sits down or stands up", ["person_sits_down", "person_stands_up"]),
    "H": ("a person rides a bicycle, reads, uses a laptop, or buys something",
          ["person_rides_bicycle", "person_reads_document", "person_interacts_with_laptop",
           "person_purchases"]),
}
TYPE_TO_GROUP = {t: g for g, (_, ts) in GROUPS.items() for t in ts}

QUESTION = (
    "These two frames were taken one second apart by a fixed security camera.\n"
    "Which of the following are happening?\n"
    + "\n".join(f"{g}. {desc[0].upper()}{desc[1:]}" for g, (desc, _) in GROUPS.items())
    + "\nAnswer with every letter that applies, separated by commas, or N if none apply."
)


def roi_box(dets_a: list, dets_b: list, w: int, h: int, pad: float = 0.25,
            min_side: int = 224) -> tuple[int, int, int, int] | None:
    ds = dets_a + dets_b
    people = [d for d in ds if d[0] == "person"]
    use = people or ds
    if not use:
        return None
    x1 = min(d[2] for d in use); y1 = min(d[3] for d in use)
    x2 = max(d[4] for d in use); y2 = max(d[5] for d in use)
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw; x2 += pad * bw; y1 -= pad * bh; y2 += pad * bh
    # never smaller than min_side: a 40 px crop gives the encoder nothing to work with
    if x2 - x1 < min_side:
        cx = (x1 + x2) / 2; x1, x2 = cx - min_side / 2, cx + min_side / 2
    if y2 - y1 < min_side:
        cy = (y1 + y2) / 2; y1, y2 = cy - min_side / 2, cy + min_side / 2
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    return x1, y1, x2, y2


def crop_jpeg(raw: bytes, box, upscale_to_short: int = 448, max_up: float = 3.0) -> bytes:
    """Crop, then upscale a small crop so its short side is ~448 px (at most 3x).

    Without the upscale a 224 px crop becomes ~64 tokens and the encoder sees the
    person at the same 14 px/patch density as in the full frame -- the crop would
    save tokens but show nothing new. The point of the ROI is to spend tokens on
    the region; the --max-pixels cap still bounds the total.
    """
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB").crop(box)
    s = min(im.size)
    k = min(max_up, max(1.0, upscale_to_short / max(s, 1)))
    if k > 1.0:
        im = im.resize((round(im.size[0] * k), round(im.size[1] * k)), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def load_dets(path: Path) -> dict[int, list]:
    out = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            r = json.loads(line)
            out[r["f"]] = r["d"]
    return out


def window_groups(acts: list[dict], w: int) -> list[str]:
    lo, hi = w * WINDOW, (w + 1) * WINDOW - 1
    return sorted({TYPE_TO_GROUP[a["type"]] for a in acts
                   if a["start"] <= hi and a["end"] >= lo and a["type"] in TYPE_TO_GROUP})


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--selection", default="results/e7/selection.json")
    p.add_argument("--index", default="data/meva/index.json")
    p.add_argument("--frame-dir", default="data/meva/frames")
    p.add_argument("--det-dir", default="data/meva/det")
    p.add_argument("--arms", default="full,roi")
    p.add_argument("--max-pixels", type=int, default=451584, help="per-frame cap")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit-clips", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=16)
    args = p.parse_args()

    from bench.data import Sample
    from bench.harness import run_arm
    from PIL import Image

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    arm.max_pixels = None          # per-frame cap applied client-side
    sel = json.load(open(args.selection))["clips"]
    if args.limit_clips:
        sel = sel[: args.limit_clips]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}

    for which in args.arms.split(","):
        samples, skipped = [], 0
        for c in sel:
            clip = c["clip"]
            meta = idx[clip]
            dets = load_dets(Path(args.det_dir) / f"{clip}.{DET_SIZE}.jsonl.gz")
            for w in range(meta["n_frames"] // WINDOW):
                fr = [w * WINDOW + o for o in VLM_OFFSETS]
                paths = [Path(args.frame_dir) / clip / f"{f:05d}.jpg" for f in fr]
                raws = [pth.read_bytes() for pth in paths]
                if which == "roi":
                    W, H = Image.open(io.BytesIO(raws[0])).size
                    box = roi_box(dets.get(fr[0], []), dets.get(fr[1], []), W, H)
                    if box is None:
                        skipped += 1
                        continue
                    raws = [crop_jpeg(r, box) for r in raws]
                gold = window_groups(meta["activities"], w)
                s = Sample(id=f"{clip}|{w}", question=QUESTION, answers=gold or ["N"])
                samples.append(s.with_frames(raws, args.max_pixels))
        print(f"\n[{which}] {len(samples)} windows"
              + (f", {skipped} skipped (nothing detected to crop)" if skipped else ""))

        summary = await run_arm(
            arm, samples, mode="closed", concurrency=args.concurrency, scorer=None,
            results_root=args.out, run_id=f"e7-vlm-{which}-{arm.id}",
            unmeasured=["latency: closed loop at fixed concurrency is for answers, "
                        "not a capacity measurement (see e7_capacity.py)",
                        "batch-composition effects on greedy decoding"],
        )
        print(f"  prompt tokens mean {summary['prompt_tokens']['mean']:.0f}   "
              f"ok {summary['n_ok']}/{len(samples)}")


if __name__ == "__main__":
    asyncio.run(main())
