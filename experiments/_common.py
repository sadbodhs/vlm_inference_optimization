"""Shared plumbing for experiment scripts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--arm", required=True, help="path to an arm YAML")
    p.add_argument("--out", default="results", help="results root")
    p.add_argument("--tag", default=None, help="label for this sweep")
    p.add_argument("--dry-run", action="store_true",
                   help="point at the mock server instead of the real stack")
    p.add_argument("--mock-url",
                   default=os.environ.get("BENCH_MOCK_URL", "http://vlm-mock:8077"))
    return p


def resolve_arm(args):
    from bench.arms import Arm

    arm = Arm.load(args.arm)
    if args.dry_run:
        arm.base_url = args.mock_url
        arm.model = "mock-vlm-7b"
        arm.id = f"{arm.id}-dryrun"
    return arm


def write_sweep(out_root: str | Path, tag: str, rows: list[dict], meta: dict) -> Path:
    d = Path(out_root) / "sweeps"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{tag}.json"
    path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, default=str))
    print(f"\nwrote {path}")
    return path


def table(rows: list[dict], cols: list[tuple[str, str]]) -> None:
    """cols = [(key, header)]"""
    widths = [max(len(h), 12) for _, h in cols]
    print("  ".join(h.ljust(w) for (_, h), w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        cells = []
        for (k, _), w in zip(cols, widths):
            v = r.get(k)
            s = f"{v:.2f}" if isinstance(v, float) else ("-" if v is None else str(v))
            cells.append(s.ljust(w))
        print("  ".join(cells))
