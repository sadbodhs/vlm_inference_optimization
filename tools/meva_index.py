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


def clip_frames(t0: str, t1: str) -> int:
    """Clip length from its name ('14-50-00', '14-55-00'). Most clips are 5 min,
    but not all -- some are 20-odd seconds -- so this is never assumed."""
    sec = lambda t: int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8])
    return int(round((sec(t1) - sec(t0)) * FPS))


def parse_activities(path: str) -> list[dict]:
    """Every activity in a KPF activities.yml -- in EITHER of MEVA's two layouts:

      - {'act': {'act2': {'person_opens_facility_door': 1.0}, ..., 'timespan': [...]}}
      - { act: { act2: {person_stands_up: 1.0}, id2: 5037..., timespan: [...], ... } }

    The first version filtered on the quoted form and silently skipped the other,
    which is 266 of the 769 clips: they indexed as "no activity at all", the corpus
    looked far emptier than it is, and the E7 "empty" bin was built from busy
    clips. So: parse any line carrying an act record, and refuse a file that has
    act lines but yields nothing.
    """
    out, n_lines = [], 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("-") or "act2" not in line:
                continue
            n_lines += 1
            rec = yaml.safe_load(line)[0]["act"]
            (name, _), = rec["act2"].items()
            if name == EMPTY_MARKER:
                n_lines -= 1          # a whole-clip "verified empty" marker, not an event
                continue
            s, e = rec["timespan"][0]["tsr0"]
            actors = [a["id1"] for a in rec.get("actors", [])]
            out.append({"type": name, "start": int(s), "end": int(e),
                        "actors": actors, "id": rec.get("id2")})
    if n_lines and len(out) != n_lines:
        raise ValueError(f"{path}: {n_lines} activity lines, {len(out)} parsed")
    return out


EMPTY_MARKER = "empty_37"


def verified_empty(path: str) -> bool:
    """Annotators mark a clip with none of the 37 activities as `empty_37` -- as an
    act record spanning the whole clip, or as a meta comment. Counting the record
    as an activity made verified-empty clips look 100% busy."""
    with open(path) as f:
        return any(EMPTY_MARKER in line for line in f)


def coverage(acts: list[dict], n_frames: int) -> float:
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
        n = clip_frames(t0, t1)
        clips.append({
            "clip": base, "date": date, "site": site, "camera": cam,
            "annotation_dir": os.path.relpath(os.path.dirname(f), args.root),
            "n_frames": n, "duration_s": n / FPS,
            "n_activities": len(acts), "coverage": round(coverage(acts, n), 4),
            "verified_empty": verified_empty(f),
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
    short = [c for c in clips if c["duration_s"] < 290]
    print(f"clips shorter than 290 s: {len(short)}")
    print("coverage (share of the 5 min with >=1 activity under way):")
    print(f"  p10 {q(.1):.2f}  p25 {q(.25):.2f}  median {q(.5):.2f}  "
          f"p75 {q(.75):.2f}  p90 {q(.9):.2f}")
    print(f"  clips with no activity at all: {sum(1 for c in clips if not c['n_activities'])}"
          f"  (verified empty by annotators: {sum(1 for c in clips if c['verified_empty'])})")
    bad = [c["clip"] for c in clips if c["verified_empty"] and c["n_activities"]]
    if bad:
        raise SystemExit(f"{len(bad)} clips marked empty but carrying activities, e.g. {bad[0]}")
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
