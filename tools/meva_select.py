#!/usr/bin/env python3
"""Pick the E7 clips: stratified by duty cycle, spread across cameras and sites.

A cascade's savings are a property of how busy the camera is, so the selection is
built to span that axis deliberately rather than inherit the corpus's mix:

  empty     no annotated activity at all
  sparse    activity under way < 10% of the clip
  moderate  10-50%
  busy      >= 50%

Within a bin: at most one clip per camera, sites visited round-robin so no bin is
all one site, deterministic for a given seed. Only full-length clips (>= 290 s)
whose video is on the public bucket are eligible.

    python3 tools/meva_select.py --per-bin 6 --out results/e7/selection.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

BINS = [("empty", None, None), ("sparse", 0.0, 0.10),
        ("moderate", 0.10, 0.50), ("busy", 0.50, 1.01)]


def bin_of(c: dict) -> str:
    if c["n_activities"] == 0:
        return "empty"
    for name, lo, hi in BINS[1:]:
        if lo <= c["coverage"] < hi:
            return name
    raise ValueError(c["clip"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--listing", default="data/meva/s3_listing.txt")
    ap.add_argument("--per-bin", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/e7/selection.json")
    args = ap.parse_args()

    idx = json.load(open(args.index))
    s3 = {}
    for line in open(args.listing):
        parts = line.split()
        if parts and parts[-1].endswith(".avi"):
            key = parts[-1]
            s3[os.path.basename(key).removesuffix(".r13.avi")] = (key, int(parts[2]))

    eligible = [c for c in idx["clips"] if c["duration_s"] >= 290 and c["clip"] in s3]
    rng = random.Random(args.seed)
    chosen = []
    for name, _, _ in BINS:
        pool = [c for c in eligible if bin_of(c) == name]
        rng.shuffle(pool)
        by_site = collections.defaultdict(list)
        for c in pool:
            by_site[c["site"]].append(c)
        sites = sorted(by_site)
        rng.shuffle(sites)
        used_cams, picked = set(), []
        while len(picked) < args.per_bin and any(by_site.values()):
            for site in sites:
                while by_site[site]:
                    c = by_site[site].pop()
                    if c["camera"] not in used_cams:
                        used_cams.add(c["camera"])
                        picked.append(c)
                        break
                if len(picked) == args.per_bin:
                    break
        if len(picked) < args.per_bin:
            raise SystemExit(f"bin {name}: only {len(picked)} eligible distinct-camera clips")
        for c in picked:
            key, size = s3[c["clip"]]
            chosen.append({"bin": name, "clip": c["clip"], "site": c["site"],
                           "camera": c["camera"], "coverage": c["coverage"],
                           "n_activities": c["n_activities"], "duration_s": c["duration_s"],
                           "s3_key": key, "bytes": size})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"seed": args.seed, "per_bin": args.per_bin,
               "bins": {n: [lo, hi] for n, lo, hi in BINS},
               "eligible": len(eligible), "clips": chosen},
              open(args.out, "w"), indent=2)

    print(f"eligible {len(eligible)} clips; chose {len(chosen)}")
    for c in chosen:
        print(f"  {c['bin']:9s} {c['site']:9s} {c['camera']:5s} cov={c['coverage']:.2f} "
              f"acts={c['n_activities']:4d}  {c['bytes']/1e6:6.0f} MB  {c['clip']}")
    print(f"total {sum(c['bytes'] for c in chosen)/1e9:.1f} GB -> {args.out}")


if __name__ == "__main__":
    main()
