#!/usr/bin/env python3
"""Freeze a video benchmark subset: clips on disk plus a manifest.

Same contract as tools/build_manifest.py, one level up in dimensionality. Frames
are NOT extracted here -- the frame count is the experiment's independent
variable, so extraction happens per-arm at run time from the stored clip.

    python3 tools/build_video_manifest.py --dataset lmms-lab/TempCompass \
        --config multi-choice --split test --n 300 --out data/tempcompass
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video-archive", default=None,
                    help="repo-relative zip of clips, e.g. tempcompass_videos.zip")
    args = ap.parse_args()

    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    ds = load_dataset(args.dataset, args.config, split=args.split)
    print(f"columns: {ds.column_names}  rows: {len(ds)}")

    out = Path(args.out)
    (out / "videos").mkdir(parents=True, exist_ok=True)

    # Clips arrive as one archive; unpack once, then keep only what the subset needs.
    if args.video_archive:
        zp = hf_hub_download(args.dataset, args.video_archive, repo_type="dataset")
        stage = out / "_unzip"
        if not stage.exists():
            print(f"unzipping {args.video_archive} ...")
            with zipfile.ZipFile(zp) as z:
                z.extractall(stage)
        index = {p.stem: p for p in stage.rglob("*") if p.suffix.lower() in
                 (".mp4", ".webm", ".mkv", ".avi")}
        print(f"  {len(index)} clips in archive")
    else:
        index = {}

    ds = ds.shuffle(seed=args.seed)
    rows, kept, missing = [], 0, 0
    for rec in ds:
        if kept >= args.n:
            break
        vid = str(rec.get("video_id") or rec.get("video") or "")
        src = index.get(vid) or index.get(Path(vid).stem)
        if src is None:
            missing += 1
            continue
        dst = out / "videos" / f"{vid}{src.suffix}"
        if not dst.exists():
            shutil.copy2(src, dst)
        rows.append({
            "id": f"{vid}-{kept}",
            "video": f"videos/{dst.name}",
            "question": rec["question"],
            "answers": [rec["answer"]],
            # The dimension is the whole point: it separates questions a single
            # frame can answer from ones that need motion.
            "dim": rec.get("dim"),
        })
        kept += 1

    (out / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    dims: dict[str, int] = {}
    for r in rows:
        dims[r["dim"]] = dims.get(r["dim"], 0) + 1
    (out / "dataset_meta.json").write_text(json.dumps(
        {"dataset": args.dataset, "config": args.config, "split": args.split,
         "n": len(rows), "seed": args.seed, "dims": dims,
         "n_missing_video": missing}, indent=2))
    shutil.rmtree(out / "_unzip", ignore_errors=True)

    print(f"wrote {len(rows)} samples to {out}/manifest.jsonl")
    print(f"  dimensions: {dims}")
    if missing:
        print(f"  WARNING: {missing} rows skipped, no clip in the archive")
    if not rows:
        raise SystemExit("no samples written")


if __name__ == "__main__":
    main()
