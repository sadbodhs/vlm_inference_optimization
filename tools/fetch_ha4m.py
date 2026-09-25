#!/usr/bin/env python3
"""Fetch the slice of HA4M that R2 needs -- never whole recordings.

HA4M (CC BY 4.0; Cicirelli et al., Scientific Data 2022) is 4.6 TB: 217 recordings
of an epicyclic-gear-train assembly, each with colour PNGs, depth, IR, point clouds
and Azure Kinect skeletons. R2 needs, per recording:

  Labels.txt                     one line per frame: frame, step (0 = idle, 1-12), sub-label
  2 colour frames per step       1 s apart (30 fps) around the step's midpoint
  2 frames of the final idle     as a "no step" sample, if the idle run is >= 31 frames
  the skeleton file of each      joint positions, incl. 2D colour-image coordinates

~26 PNGs x 3.15 MB x 217 recordings ~ 18 GB. Files are verified against their
WebDAV content length and skipped when already present, so a rerun resumes.

    docker/run_harness.sh python3 tools/fetch_ha4m.py            # -> data/ha4m/
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHARE = "j5BZvG1sXPzh6x1"                 # the README's public cloud.cnr.it share
DAV = "https://cloud.cnr.it/owncloud/public.php/webdav"
AUTH = {"Authorization": "Basic " + b64encode(f"{SHARE}:".encode()).decode()}
GAP = 15                                   # frames either side of the midpoint: 1 s apart


def dav(path: str, depth: int = 1) -> list[tuple[str, int]]:
    body = (b'<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop>'
            b"<d:getcontentlength/></d:prop></d:propfind>")
    req = urllib.request.Request(DAV + urllib.parse.quote(path), data=body, method="PROPFIND",
                                 headers={**AUTH, "Depth": str(depth), "Content-Type": "application/xml"})
    for attempt in range(5):
        try:
            x = urllib.request.urlopen(req, timeout=120).read().decode()
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    out = []
    for r in re.findall(r"<d:response>(.*?)</d:response>", x, re.S)[1:]:
        name = urllib.parse.unquote(re.search(r"<d:href>([^<]+)</d:href>", r).group(1)).rstrip("/").split("/")[-1]
        n = re.search(r"<d:getcontentlength>(\d+)<", r)
        out.append((name, int(n.group(1)) if n else -1))
    return out


def get(path: str, dest: Path, size: int | None = None) -> int:
    if dest.exists() and (size is None or dest.stat().st_size == size):
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(DAV + urllib.parse.quote(path), headers=AUTH)
    for attempt in range(5):
        try:
            data = urllib.request.urlopen(req, timeout=300).read()
            if size is not None and len(data) != size:
                raise IOError(f"{path}: {len(data)} bytes, expected {size}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return len(data)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    return 0


def pick_frames(labels: list[tuple[int, int]]) -> list[dict]:
    """Two frames per step segment (and per trailing idle run)."""
    samples, pos = [], 0
    segs = [(k, len(list(g))) for k, g in itertools.groupby(l for _, l in labels)]
    for i, (step, n) in enumerate(segs):
        start, end = pos, pos + n - 1
        pos += n
        trailing_idle = step == 0 and i == len(segs) - 1
        if step == 0 and not trailing_idle:
            continue
        if n < 2 * GAP + 1:
            if step == 0:
                continue
            a, b = start, end              # a short step: its first and last frame
        else:
            m = (start + end) // 2
            a, b = m - GAP, m + GAP
        samples.append({"step": step, "seg_start": start, "seg_end": end,
                        "frames": [labels[a][0], labels[b][0]]})
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ha4m")
    ap.add_argument("--limit", type=int, default=0, help="recordings, for a trial")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    recs = sorted(n for n, _ in dav("/") if re.fullmatch(r"IDU\d+V\d+", n))
    if args.limit:
        recs = recs[:args.limit]
    print(f"{len(recs)} recordings", flush=True)
    manifest, got, t0 = [], 0, time.time()
    pool = ThreadPoolExecutor(args.workers)
    for k, rec in enumerate(recs):
        d = out / rec
        get(f"/{rec}/Labels.txt", d / "labels.txt")
        labels = [(int(a), int(b)) for a, b, *_ in
                  (l.split() for l in (d / "labels.txt").read_text().splitlines() if l.strip())]
        chosen = pick_frames(labels)
        need = {f for s in chosen for f in s["frames"]}
        files = {}
        for kind in ("Color", "Skeletons"):
            sub = [n for n, _ in dav(f"/{rec}/{kind}/")]
            if len(sub) != 1:
                print(f"  {rec}: {kind} has {len(sub)} subfolders, skipped", flush=True)
                files[kind] = {}
                continue
            idx = {}
            for name, size in dav(f"/{rec}/{kind}/{sub[0]}/"):
                m = re.match(r"FrameID(\d+)_", name)
                if m and int(m.group(1)) in need:
                    idx[int(m.group(1))] = (f"/{rec}/{kind}/{sub[0]}/{name}", size)
            files[kind] = idx
        jobs = []
        for f in need:
            if f in files.get("Color", {}):
                p, s = files["Color"][f]
                jobs.append(pool.submit(get, p, d / "color" / f"{f:06d}.png", s))
            if f in files.get("Skeletons", {}):
                p, s = files["Skeletons"][f]
                jobs.append(pool.submit(get, p, d / "skel" / f"{f:06d}.txt", s))
        got += sum(j.result() for j in jobs)
        for s in chosen:
            ok = all((d / "color" / f"{f:06d}.png").exists() for f in s["frames"])
            manifest.append({"rec": rec, "subject": rec[:6], **s, "complete": ok})
        if k % 10 == 0 or k == len(recs) - 1:
            print(f"  {k+1}/{len(recs)} {rec}: {len(chosen)} samples; "
                  f"{got/1e9:.2f} GB so far, {(time.time()-t0)/60:.1f} min", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"samples {len(manifest)}, complete {sum(m['complete'] for m in manifest)}; "
          f"downloaded {got/1e9:.2f} GB", flush=True)
    sys.exit(0 if all(m["complete"] for m in manifest) else 1)


if __name__ == "__main__":
    main()
