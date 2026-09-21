#!/usr/bin/env python3
"""Render sweep JSON to PNG. One chart per experiment, no styling cleverness.

    python3 experiments/plot.py results/sweeps/e2-ttft-*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FG, ACCENT, WARN = "#1a1a1a", "#c1440e", "#888888"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=FG, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=FG)
    ax.set_ylabel(ylabel, fontsize=9, color=FG)
    ax.grid(alpha=0.18, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=FG)


def plot_e0(d, out):
    rows = [r for r in d["rows"] if r["verdict"] != "client-bound"]
    bad = [r for r in d["rows"] if r["verdict"] == "client-bound"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot([r["offered_qps"] for r in rows], [r["achieved_qps"] for r in rows],
            "o-", color=ACCENT, label="achieved")
    lim = max([r["offered_qps"] for r in d["rows"]] or [1])
    a1.plot([0, lim], [0, lim], "--", color=WARN, linewidth=1, label="offered (y=x)")
    if bad:
        a1.scatter([r["offered_qps"] for r in bad], [r["achieved_qps"] for r in bad],
                   marker="x", color=WARN, label="client-bound (discarded)")
    a1.legend(fontsize=8, frameon=False)
    _style(a1, "E0 · throughput vs offered load", "offered QPS", "achieved req/s")

    a2.plot([r["offered_qps"] for r in rows], [r["ttft_p99_ms"] for r in rows],
            "o-", color=ACCENT)
    a2.set_yscale("log")
    _style(a2, "E0 · TTFT p99 (queueing knee)", "offered QPS", "TTFT p99 (ms, log)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e2(d, out):
    rows = [r for r in d["rows"] if r.get("vision_tokens")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = [r["vision_tokens"] for r in rows]
    ax.plot(x, [r["ttft_p50_ms"] for r in rows], "o-", color=ACCENT, label="TTFT p50")
    ax.fill_between(x, [r["ttft_p50_ms"] for r in rows],
                    [r["ttft_p95_ms"] for r in rows], alpha=0.12, color=ACCENT)
    if fit := d["meta"].get("fit"):
        xs = [min(x), max(x)]
        ax.plot(xs, [fit["intercept_ms"] + fit["slope_ms_per_token"] * v for v in xs],
                "--", color=WARN, linewidth=1,
                label=f"fit: {fit['intercept_ms']:.0f}ms + "
                      f"{fit['slope_ms_per_token']*1000:.1f}ms/1k tok")
        ax.axhline(fit["intercept_ms"], color=WARN, linewidth=0.7, alpha=0.5)
    ax.set_ylim(bottom=0)  # the intercept is the point; don't crop it off
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "E2 · TTFT vs vision tokens", "vision tokens", "TTFT (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e3(d, out):
    rows = [r for r in d["rows"] if r.get("accuracy") is not None]
    if not rows:
        print("  e3: no accuracy in this sweep, skipping Pareto plot")
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot([r["ttft_p50_ms"] for r in rows], [r["accuracy"] for r in rows],
            "o-", color=ACCENT)
    for r in rows:
        ax.annotate(f"{r['max_edge_sq']}px", (r["ttft_p50_ms"], r["accuracy"]),
                    textcoords="offset points", xytext=(5, -9), fontsize=7, color=WARN)
    _style(ax, f"E3 · accuracy vs TTFT  ({d['meta'].get('dataset')})",
           "TTFT p50 (ms)", f"accuracy ({d['meta'].get('scorer')})")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e1(d, out):
    r = d["rows"][0]
    if not r.get("roofline_tok_s"):
        print("  e1: no weight_bytes in the arm, no roofline to draw")
        return
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(["measured", "roofline"], [r["decode_tok_s"], r["roofline_tok_s"]],
           color=[ACCENT, WARN], width=0.55)
    ax.axhline(r["roofline_tok_s"], color=WARN, linestyle="--", linewidth=1)
    pctv = r.get("pct_of_roofline")
    if pctv:
        ax.annotate(f"{pctv:.0f}% of roofline", (0, r["decode_tok_s"]),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=9, color=FG)
    _style(ax, f"E1 · single-stream decode ({r['arm']}, "
               f"{r.get('decode_weight_GB') or r['weight_GB']:.1f} GB re-read per token)",
           "", "tokens/s")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


PLOTS = {"e0-saturation": plot_e0, "e1-roofline": plot_e1, "e2-ttft-vs-tokens": plot_e2,
         "e3-token-budget": plot_e3}


def main(paths: list[str]) -> None:
    for p in paths:
        d = json.loads(Path(p).read_text())
        exp = d["meta"].get("experiment")
        fn = PLOTS.get(exp)
        if not fn:
            print(f"  no plotter for {exp!r} ({p})")
            continue
        out = Path(p).with_suffix(".png")
        fn(d, out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(p) for p in Path("results/sweeps").glob("*.json")])
