#!/usr/bin/env python3
"""Bootstrap confidence intervals for per-run accuracy, and test pairwise differences.

A single accuracy number from n=100 invites claims the sample size cannot support.
This re-scores each run per sample, bootstraps the mean, and reports whether two
budgets actually differ -- paired on sample id, since the same documents are used
at every budget and a paired test is far more sensitive than comparing two CIs.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.scorers import SCORERS  # noqa: E402


def per_sample(run_dir: Path, gold: dict, scorer) -> dict[str, float]:
    out = {}
    for line in (run_dir / "outputs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = gold.get(str(r["sample_id"]))
        if g and g.get("answers"):
            out[str(r["sample_id"])] = scorer(r["text"], g["answers"])
    return out


def boot_ci(xs: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(xs)
    means = sorted(sum(rng.choices(xs, k=k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--scorer", default="anls")
    args = ap.parse_args()

    gold = {}
    for line in Path(args.manifest).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            gold[str(r["id"])] = r
    scorer = SCORERS[args.scorer]

    series = {}
    for d in args.runs:
        p = Path(d)
        series[p.name.split("-")[-1]] = per_sample(p, gold, scorer)

    print(f"{'budget':<12} {'n':>4} {'mean':>7}   95% CI")
    for name, s in series.items():
        xs = list(s.values())
        lo, hi = boot_ci(xs)
        print(f"{name:<12} {len(xs):>4} {sum(xs)/len(xs):>7.3f}   [{lo:.3f}, {hi:.3f}]  ±{(hi-lo)/2:.3f}")

    names = list(series)
    print(f"\npaired differences (same documents at both budgets):")
    print(f"{'A':<12} {'B':<12} {'Δ(A-B)':>8}   95% CI          verdict")
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(series[a]) & set(series[b]))
            diffs = [series[a][k] - series[b][k] for k in common]
            lo, hi = boot_ci(diffs)
            mean = sum(diffs) / len(diffs)
            sig = "DIFFERENT" if (lo > 0 or hi < 0) else "indistinguishable"
            print(f"{a:<12} {b:<12} {mean:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
