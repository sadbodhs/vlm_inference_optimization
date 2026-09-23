#!/usr/bin/env python3
"""Index the Kitware MEVA activity annotations into one JSON the cascade can use.

The Kitware set (769 clips) is annotated *exhaustively* for all 37 ActEV activity
types, audited the same way as the sequestered evaluation set. That property is
what the cascade needs: a gate "missing" an activity is only measurable if every
activity is labelled -- an unlabelled event that the gate skips would score as a
correct skip. The smaller NIST-format set (65 clips, ~120 activities) is a sample
and is deliberately not used.

KPF activity lines are YAML flow mappings, one per line:
  - {'act': {'act2': {'person_opens_facility_door': 1.0}, 'actors': [...],
             'id2': 9, 'timespan': [{'tsr0': [982, 1042]}]}}
Parsed with PyYAML one line at a time: the files are large and a whole-document
parse is slow and memory-hungry for no benefit.

    python3 tools/meva_index.py --root data/meva/meva-data-repo --out data/meva/index.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st

import yaml

FPS = 30.0            # MEVA ground cameras are 30 fps; file-index confirms per clip
CLIP_FRAMES = 9000    # 5 minutes


def parse_activities(path: str) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("- {'act'"):
                continue
            rec = yaml.safe_load(line)[0]["act"]
            (name, _), = rec["act2"].items()
            s, e = rec["timespan"][0]["tsr0"]
            actors = [a["id1"] for a in rec.get("actors", [])]
            out.append({"type": name, "start": int(s), "end": int(e),
                        "actors": actors, "id": rec.get("id2")})
    return out


def coverage(acts: list[dict], n_frames: int = CLIP_FRAMES) -> float:
    """Fraction of the clip during which at least one activity is under way."""
    covered = [False] * n_frames
    for a in acts:
        for fr in range(max(0, a["start"]), min(n_frames, a["end"] + 1)):
            covered[fr] = True
    return sum(covered) / n_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/meva/meva-data-repo")
    ap.add_argument("--out", default="data/meva/index.json")
    args = ap.parse_args()

    ann = os.path.join(args.root, "annotation/DIVA-phase-2/MEVA/kitware")
    files = sorted(glob.glob(f"{ann}/**/*.activities.yml", recursive=True))
    clips = []
    for f in files:
        base = os.path.basename(f).removesuffix(".activities.yml")
        date, t0, t1, site, cam = base.split(".")[:5]
        acts = parse_activities(f)
        clips.append({
            "clip": base, "date": date, "site": site, "camera": cam,
            "annotation_dir": os.path.relpath(os.path.dirname(f), args.root),
            "n_activities": len(acts), "coverage": round(coverage(acts), 4),
            "activities": acts,
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"source": "MEVA Kitware KPF annotations (CC BY 4.0)",
                   "fps": FPS, "clips": clips}, f)

    # ── profile: what the corpus looks like, printed so selection is justified ──
    types = collections.Counter(a["type"] for c in clips for a in c["activities"])
    covs = sorted(c["coverage"] for c in clips)
    q = lambda p: covs[int(p * (len(covs) - 1))]
    print(f"clips {len(clips)}   activities {sum(types.values())}   types {len(types)}")
    print("sites", dict(collections.Counter(c["site"] for c in clips)))
    print("coverage (share of the 5 min with >=1 activity under way):")
    print(f"  p10 {q(.1):.2f}  p25 {q(.25):.2f}  median {q(.5):.2f}  "
          f"p75 {q(.75):.2f}  p90 {q(.9):.2f}")
    print(f"  clips with no activity at all: {sum(1 for c in clips if not c['n_activities'])}")
    durs = sorted((a["end"] - a["start"]) / FPS for c in clips for a in c["activities"])
    print(f"activity duration (s): p10 {durs[len(durs)//10]:.1f}  "
          f"median {st.median(durs):.1f}  p90 {durs[9*len(durs)//10]:.1f}")
    print("types:")
    for k, v in types.most_common():
        print(f"  {v:6d}  {k}")
    bycam = collections.defaultdict(list)
    for c in clips:
        bycam[(c["site"], c["camera"])].append(c["coverage"])
    print(f"cameras {len(bycam)} (most-annotated first):")
    for (site, cam), v in sorted(bycam.items(), key=lambda kv: -len(kv[1]))[:30]:
        print(f"  {site:9s} {cam:5s} clips={len(v):3d}  median coverage={st.median(v):.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
