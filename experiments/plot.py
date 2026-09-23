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


# plot_e6 is defined further down, so the registry is populated lazily at call
# time rather than at import -- see _plotter().
PLOTS = {"e0-saturation": plot_e0, "e1-roofline": plot_e1, "e2-ttft-vs-tokens": plot_e2,
         "e3-token-budget": plot_e3}


def _plotter(name):
    if name == "e6-temporal-budget":
        return plot_e6
    return PLOTS.get(name)


def main(paths: list[str]) -> None:
    for p in paths:
        d = json.loads(Path(p).read_text())
        exp = d["meta"].get("experiment")
        fn = _plotter(exp)
        if not fn:
            print(f"  no plotter for {exp!r} ({p})")
            continue
        out = Path(p).with_suffix(".png")
        fn(d, out)
        print(f"  wrote {out}")


# ── comparison plots: several sweeps on one pair of axes ─────────────────────
# These answer questions no single sweep can: does the budget depend on the task,
# does the frame count matter differently per question type, how do two stacks or
# two model generations differ. Each takes a list of sweep files.

SERIES = ["#c1440e", "#1f4e8c", "#3f7d5a", "#8a5cb8", "#b0892a"]


def plot_task_frontier(paths, out):
    """Accuracy vs vision tokens, one line per task (each on its own metric)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for i, p in enumerate(paths):
        d = json.loads(Path(p).read_text())
        rows = [r for r in d["rows"] if r.get("accuracy") is not None]
        if not rows:
            continue
        name = d["meta"].get("dataset", p)
        label = Path(str(name)).parts[1] if "data/" in str(name) else Path(p).stem
        peak = max(r["accuracy"] for r in rows)
        x = [r["prompt_tokens"] for r in rows]
        # normalised to each task's own peak: the metrics differ, so only the
        # SHAPE of each curve is comparable across tasks
        y = [100 * r["accuracy"] / peak for r in rows]
        ax.plot(x, y, "o-", color=SERIES[i % len(SERIES)], label=f"{label} (peak {peak:.3f})")
    ax.axhline(100, color=WARN, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "Per-task frontier · accuracy as % of that task's own peak",
           "vision tokens (log)", "% of own peak")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e6(d, out):
    """Per-dimension accuracy vs frame count, against the chance floor."""
    rows = d["rows"]
    dims = d["meta"].get("dims") or []
    chance = d["meta"].get("chance_level", 1 / 3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for i, dim in enumerate(dims):
        xs = [r["frames"] for r in rows if r.get(dim) is not None]
        ys = [r[dim] for r in rows if r.get(dim) is not None]
        if xs:
            a1.plot(xs, ys, "o-", color=SERIES[i % len(SERIES)], label=dim)
    a1.axhline(chance, color=WARN, lw=1, ls="--", label=f"chance ({chance:.2f})")
    a1.set_xscale("log", base=2)
    a1.set_xticks([r["frames"] for r in rows])
    a1.set_xticklabels([str(r["frames"]) for r in rows])
    a1.legend(fontsize=8, frameon=False)
    _style(a1, "E6 · accuracy by question type", "frames per clip", "accuracy")

    a2.plot([r["frames"] for r in rows], [r["ttft_p50_ms"] for r in rows],
            "o-", color=ACCENT)
    a2.set_xscale("log", base=2)
    a2.set_xticks([r["frames"] for r in rows])
    a2.set_xticklabels([str(r["frames"]) for r in rows])
    _style(a2, "E6 · what frames cost", "frames per clip", "TTFT p50 (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_compare_e3(paths, labels, title, out):
    """Two arms' accuracy/TTFT frontiers overlaid — stacks, or model generations."""
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for i, (p, lab) in enumerate(zip(paths, labels)):
        d = json.loads(Path(p).read_text())
        rows = [r for r in d["rows"] if r.get("accuracy") is not None]
        ax.plot([r["ttft_p50_ms"] for r in rows], [r["accuracy"] for r in rows],
                "o-", color=SERIES[i % len(SERIES)], label=lab)
        for r in rows:
            ax.annotate(f"{r['prompt_tokens']:.0f}", (r["ttft_p50_ms"], r["accuracy"]),
                        textcoords="offset points", xytext=(4, -9), fontsize=6, color=WARN)
    ax.legend(fontsize=8, frameon=False)
    _style(ax, title, "TTFT p50 (ms)", "accuracy (ANLS)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def comparisons(root="results/sweeps", out_dir="docs/img"):
    """Build the cross-sweep figures the comparison pages need."""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [root / f"e3-{t}.json" for t in ("docvqa", "chartqa", "textvqa", "vqav2")]
    tasks = [p for p in tasks if p.exists()]
    if len(tasks) >= 2:
        plot_task_frontier([str(p) for p in tasks], out_dir / "task-frontier.png")
        print(f"  wrote {out_dir/'task-frontier.png'}")

    pairs = [
        (["e3-budget-B0_vllm_awq_clean.json", "e3-budget-C0_sglang_awq_clean.json"],
         ["vLLM", "SGLang"], "vLLM vs SGLang · accuracy against TTFT", "stacks.png"),
        (["e3-budget-B0_vllm_awq_clean.json", "e3-budget-Q3_vllm_awq_clean.json"],
         ["Qwen2.5-VL-7B", "Qwen3-VL-8B"], "Model generations · accuracy against TTFT",
         "models.png"),
    ]
    for files, labels, title, name in pairs:
        paths = [root / f for f in files]
        if all(p.exists() for p in paths):
            plot_compare_e3([str(p) for p in paths], labels, title, out_dir / name)
            print(f"  wrote {out_dir/name}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--comparisons":
        comparisons()
    else:
        main(args or [str(p) for p in Path("results/sweeps").glob("*.json")])
