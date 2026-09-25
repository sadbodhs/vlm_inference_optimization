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
    _style(ax, f"E1 · single-stream decode, {r['arm']}\n"
               f"{r.get('decode_weight_GB') or r['weight_GB']:.2f} GB re-read per token",
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

# Categorical order is fixed: a task keeps its colour on every figure. Validated with
# the dataviz palette checker (lightness band, chroma floor, CVD and normal-vision
# separation, contrast) -- the previous blue was too dark and the green too grey.
SERIES = ["#c1440e", "#2a64b4", "#1f8a52", "#8a5cb8", "#b0892a"]
TASKS = ("docvqa", "chartqa", "textvqa", "vqav2")
TASK_LABEL = {"docvqa": "DocVQA", "chartqa": "ChartQA", "textvqa": "TextVQA",
              "vqav2": "VQAv2"}


def plot_task_frontier(paths, out):
    """Accuracy vs vision tokens, one line per task (each on its own metric)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for i, p in enumerate(paths):
        d = json.loads(Path(p).read_text())
        rows = [r for r in d["rows"] if r.get("accuracy") is not None]
        if not rows:
            continue
        name = d["meta"].get("dataset", p)
        task = Path(str(name)).parts[1] if "data/" in str(name) else Path(p).stem
        # colour follows the task, not its position in the list
        color = SERIES[TASKS.index(task)] if task in TASKS else SERIES[i % len(SERIES)]
        peak = max(r["accuracy"] for r in rows)
        x = [r["prompt_tokens"] for r in rows]
        # normalised to each task's own peak: the metrics differ, so only the
        # SHAPE of each curve is comparable across tasks
        y = [100 * r["accuracy"] / peak for r in rows]
        ax.plot(x, y, "o-", color=color, lw=2,
                label=f"{TASK_LABEL.get(task, task)} (peak {peak:.3f})")
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



def _task_rows(root):
    """{task: rows-with-accuracy} for whichever per-task sweeps exist."""
    out = {}
    for t in TASKS:
        f = Path(root) / f"e3-{t}.json"
        if f.exists():
            rows = [r for r in json.loads(f.read_text())["rows"]
                    if r.get("accuracy") is not None]
            if rows:
                out[t] = rows
    return out


def plot_headline(root, out):
    """The thesis in one picture: what each task keeps as TTFT is cut."""
    data = _task_rows(root)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for t, rows in data.items():
        c = SERIES[TASKS.index(t)]
        peak = max(r["accuracy"] for r in rows)
        x = [r["ttft_p50_ms"] for r in rows]
        y = [100 * r["accuracy"] / peak for r in rows]
        ax.plot(x, y, "-", color=c, lw=2, label=TASK_LABEL[t])
        ax.plot(x, y, "o", color=c, ms=7, mec="white", mew=1.5)
        # direct label at the cheapest point: that is where the tasks separate
        ax.annotate(TASK_LABEL[t], (x[0], y[0]), textcoords="offset points",
                    xytext=(-8, 0), ha="right", va="center", fontsize=8, color=FG)
    ax.axhline(100, color=WARN, lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(25, 900)
    ax.set_ylim(25, 105)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "What each task keeps as the image is shrunk\n"
               "Qwen2.5-VL-7B-AWQ, one RTX 3090, 400 samples per point",
           "time to first token, p50 (ms, log)", "% of the task's own peak accuracy")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e4(root, out):
    """Live video: staleness as streams grow, and what happens past the knee."""
    root = Path(root)
    streams = []
    for n in (1, 2, 4, 8):
        f = root / f"e4-video-B0-s{n}.json"
        if f.exists():
            streams.append(json.loads(f.read_text())["rows"][0])
    load = []
    for iv in ("1.0", "0.5", "0.25", "0.125"):
        f = root / f"e4-open-iv{iv}.json"
        if f.exists():
            load.append(json.loads(f.read_text())["rows"][0])
    if not streams or not load:
        print("  e4: sweeps missing, skipping")
        return
    slo = 2000
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 3.9))

    n = [r["streams"] for r in streams]
    a1.plot(n, [r["staleness_p50_ms"] for r in streams], "o-", color=SERIES[1], lw=2,
            label="p50")
    a1.plot(n, [r["staleness_p95_ms"] for r in streams], "o--", color=SERIES[1], lw=1.5,
            alpha=0.6, label="p95")
    a1.axhline(slo, color=ACCENT, lw=1, ls=":")
    a1.annotate("2 s freshness budget", (1, slo), textcoords="offset points",
                xytext=(0, 4), fontsize=7, color=FG)
    a1.set_xscale("log", base=2); a1.set_xticks(n); a1.set_xticklabels(map(str, n))
    a1.set_ylim(0, 2400)
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    _style(a1, "Staleness, 1 fps per stream", "RTSP streams", "answer age (ms)")

    d = [r["demand_per_s"] for r in load]
    a2.plot([min(d), max(d)], [min(d), max(d)], ":", color=WARN, lw=1,
            label="completed = demanded")
    a2.plot(d, [r["analyses_per_s"] for r in load], "o-", color=SERIES[1], lw=2,
            label="completed")
    a2.axhline(8, color=ACCENT, lw=1, ls=":", label="usable (fresh) capacity")
    a2.legend(fontsize=7, frameon=False, loc="lower right")
    a2.set_xscale("log", base=2); a2.set_xticks(d); a2.set_xticklabels([f"{x:.0f}" for x in d])
    a2.set_ylim(0, 18)
    _style(a2, "Throughput keeps rising past the knee", "analyses demanded /s",
           "analyses completed /s")

    a3.bar([str(int(x)) for x in d], [100 * r["fresh_fraction"] for r in load],
           color=SERIES[1], width=0.55)
    for i, r in enumerate(load):
        a3.annotate(f"{100*r['fresh_fraction']:.0f}%", (i, 100 * r["fresh_fraction"]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8,
                    color=FG)
    a3.set_ylim(0, 110)
    _style(a3, "…while every answer goes stale", "analyses demanded /s",
           "answers under 2 s old (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


# Historical bases for the roofline correction (docs/corrections.md, e1-roofline.md).
# Kept as constants: the first two no longer exist in any arm file, by design.
HBM_BW_GB_S = 936.0
ROOFLINE_BASES = [("total weights (first basis)", 7.161),
                  ("LLM weights, estimated", 5.807),
                  ("LLM weights, exact", 5.571)]


def plot_corrections(root, out):
    """Three bugs next to their fixes, each on its own axes."""
    root = Path(root)
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.5, 3.9))

    # 1 · VQAv2 scored against stringified dicts
    bad = root / "broken" / "e3-vqav2-dict-golds.json"
    good = root / "e3-vqav2.json"
    if bad.exists() and good.exists():
        for f, lab, c, st in ((good, "fixed", SERIES[3], "o-"),
                              (bad, "gold = str(dict)", WARN, "s--")):
            rows = json.loads(f.read_text())["rows"]
            a1.plot([r["prompt_tokens"] for r in rows], [r["accuracy"] for r in rows],
                    st, color=c, lw=2, label=lab)
        a1.set_ylim(-0.05, 1.0)
        a1.legend(fontsize=8, frameon=False, loc="center right")
    _style(a1, "VQAv2: 0.000 at every budget", "vision tokens", "exact-match accuracy")

    # 2 · E4 closed loop measured its own pacing
    closed, opened = [], []
    for iv in ("1.0", "0.5", "0.25", "0.125"):
        fc, fo = root / f"e4-load-iv{iv}.json", root / f"e4-open-iv{iv}.json"
        if fc.exists() and fo.exists():
            closed.append(json.loads(fc.read_text())["rows"][0])
            opened.append(json.loads(fo.read_text())["rows"][0])
    if opened:
        d = [r["demand_per_s"] for r in opened]
        a2.plot(d, [r["analyses_per_s"] for r in opened], "o-", color=SERIES[1], lw=2,
                label="open loop (fixed)")
        a2.plot(d, [r["analyses_per_s"] for r in closed], "s--", color=WARN, lw=2,
                label="closed loop (bug)")
        a2.set_xscale("log", base=2); a2.set_xticks(d)
        a2.set_xticklabels([f"{x:.0f}" for x in d])
        a2.set_ylim(0, 16)
        a2.legend(fontsize=8, frameon=False, loc="lower right")
    _style(a2, "E4: a flat line that was the harness", "analyses demanded /s",
           "analyses completed /s")

    # 3 · roofline basis
    e1 = root / "e1-roofline-B0_vllm_awq_clean.json"
    if e1.exists():
        meas = json.loads(e1.read_text())["rows"][0]["decode_tok_s"]
        a3.bar(["measured"], [meas], color=ACCENT, width=0.45)
        styles = [(WARN, ":"), (WARN, "--"), (FG, "-")]
        for (lab, gb), (c, ls) in zip(ROOFLINE_BASES, styles):
            ceil = HBM_BW_GB_S / gb
            a3.axhline(ceil, color=c, lw=1.2, ls=ls,
                       label=f"{lab}: {ceil:.0f} tok/s → {100*meas/ceil:.0f}%")
        a3.set_xlim(-0.5, 2.8)
        a3.set_ylim(0, 190)
        a3.legend(fontsize=7, frameon=False, loc="lower right", title="ceiling basis",
                  title_fontsize=7)
    _style(a3, "E1: 114% of a ceiling cannot exist", "", "decode tokens/s")
    fig.tight_layout()
    fig.savefig(out, dpi=160)



# ── E7 · the cascade ─────────────────────────────────────────────────────────
E7_GATES = ("presence", "person", "motion", "person-motion")
E7_BINS = ("empty", "sparse", "moderate", "busy")


def _e7(path):
    return json.loads(Path(path).read_text())


def plot_e7_gates(gates_json, out):
    """Left: what each gate sends vs what it keeps. Right: savings by scene."""
    d = _e7(gates_json)
    calls, rec = d["calls"], d["recall"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))
    for i, size in enumerate((640, 1280)):
        c = SERIES[1] if size == 1280 else WARN
        xs = [100 * calls[f"{size}|{g}|all"]["rate"] for g in E7_GATES]
        ys = [100 * rec[f"{size}|{g}|all"]["rate"] for g in E7_GATES]
        a1.plot(xs, ys, "o", color=c, ms=8, mec="white", mew=1.5, label=f"YOLOv8s @ {size}")
        for g, x, y in zip(E7_GATES, xs, ys):
            if size == 1280:
                a1.annotate(g, (x, y), textcoords="offset points", xytext=(7, -3),
                            fontsize=8, color=FG)
    a1.plot([100], [100], "s", color=FG, ms=7, label="dense (every window)")
    a1.set_xlim(25, 108); a1.set_ylim(92, 100.8)
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    _style(a1, "What a gate sends, and what it keeps", "windows sent to the VLM (%)",
           "annotated activities covered (%)")

    size, w = 1280, 0.2
    for j, g in enumerate(E7_GATES):
        ys = [100 * (1 - calls[f"{size}|{g}|{b}"]["rate"]) for b in E7_BINS]
        a2.bar([k + (j - 1.5) * w for k in range(len(E7_BINS))], ys, width=w * 0.92,
               color=SERIES[j], label=g)
    a2.set_xticks(range(len(E7_BINS)))
    a2.set_xticklabels(["empty\n(verified)", "sparse\n(<10% active)", "moderate\n(10-50%)",
                        "busy\n(>=50%)"], fontsize=8)
    a2.set_ylim(0, 105)
    a2.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    _style(a2, "VLM calls saved depend on the scene (YOLO @ 1280)", "",
           "calls saved vs dense (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e7_detector(gates_json, out):
    """Detector recall on annotated people, by box height, at both input sizes."""
    d = _e7(gates_json)["detector_recall"]
    bins = ["0-25", "25-50", "50-100", "100-200", "200+"]
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for size, c in ((640, WARN), (1280, SERIES[1])):
        ys = [100 * (d.get(f"{size}|person|{b}", {}).get("rate") or 0) for b in bins]
        ax.plot(range(len(bins)), ys, "o-", color=c, lw=2, label=f"YOLOv8s @ {size}")
    ns = [d.get(f"1280|person|{b}", {}).get("n", 0) for b in bins]
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{b}\nn={n:,}" for b, n in zip(bins, ns)], fontsize=8)
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "Does the detector see the people doing things?",
           "annotated actor height in the 1080p frame (px)", "detected (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


E7_GROUP_SHORT = {"A": "gets into / out of a vehicle, doors, trunk",
                  "B": "vehicle starts, stops, turns, reverses",
                  "C": "walks through a doorway, opens a door",
                  "D": "picks up, puts down, carries an object",
                  "E": "uses a phone",
                  "F": "people talk to / touch each other",
                  "G": "sits down / stands up",
                  "H": "bicycle, reading, laptop, buying"}


def plot_e7_groups(cascades_json, out, groups_desc=None, min_n=20):
    """Chance-corrected recognition per activity group, full frame vs ROI crop.
    Groups with fewer than `min_n` instances are drawn hollow: too few to read."""
    rows = [r for r in _e7(cascades_json)["rows"]
            if r["gate"] == "dense" and r["bin"] == "all" and r["size"] == 1280]
    by = {r["arm"]: r["per_group"] for r in rows}
    groups = sorted(by["full"], key=lambda g: (by["full"][g]["n"] < min_n, -by["full"][g]["lift"]))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    h = 0.38
    for j, (arm, c) in enumerate((("full", SERIES[1]), ("roi", SERIES[0]))):
        if arm not in by:
            continue
        for k, g in enumerate(groups):
            small = by["full"][g]["n"] < min_n
            ax.barh(k + (j - 0.5) * h, 100 * by[arm][g]["lift"], height=h * 0.9,
                    color="none" if small else c, edgecolor=c, lw=1.2,
                    label=(("full frame" if arm == "full" else "ROI crop") if k == 0 else None))
    ax.axvline(0, color=FG, lw=0.8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{E7_GROUP_SHORT[g]}  (n={by['full'][g]['n']})"
                        + ("  too few" if by["full"][g]["n"] < min_n else "")
                        for g in groups], fontsize=8)
    ax.invert_yaxis()
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "What Qwen2.5-VL-7B can see at surveillance distance",
           "recognised above chance (percentage points)", "")
    fig.tight_layout()
    fig.savefig(out, dpi=160)



def plot_e7_cameras(cascades_json, out, size=1280):
    """Cameras per 3090, and what each camera's GPU budget is spent on."""
    rows = {(r["gate"], r["arm"]): r for r in _e7(cascades_json)["rows"]
            if r["bin"] == "all" and r["size"] == size}
    configs = [("dense", "full"), ("motion", "full"), ("person-motion", "full"),
               ("dense", "roi"), ("motion", "roi"), ("person-motion", "roi")]
    labels = [f"{g}\n{'full frame' if a == 'full' else 'ROI crop'}\n"
              f"{100*rows[(g, a)]['gate_recall']:.0f}% covered" for g, a in configs]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.3))
    cams = [rows[k]["cameras_per_gpu"] or 0 for k in configs]
    cols = [SERIES[1] if a == "full" else SERIES[0] for _, a in configs]
    a1.bar(range(len(configs)), cams, color=cols, width=0.62)
    for i, k in enumerate(configs):
        r = rows[k]
        a1.annotate(f"{cams[i]:.1f}", (i, cams[i]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=9, color=FG)
    a1.set_xticks(range(len(configs))); a1.set_xticklabels(labels, fontsize=7.5)
    _style(a1, f"Cameras per RTX 3090 (YOLOv8s @ {size}, answers < 2 s old)", "",
               "cameras")

    det = [100 * rows[k]["detector_share"] for k in configs]
    vlm = [100 * (rows[k]["vlm_share"] or 0) for k in configs]
    a2.bar(range(len(configs)), vlm, color=cols, width=0.62, label="VLM")
    a2.bar(range(len(configs)), det, bottom=vlm, color=WARN, width=0.62,
           label="detector (PyTorch fp32, 5 fps)")
    a2.set_xticks(range(len(configs))); a2.set_xticklabels(labels, fontsize=7.5)
    a2.set_ylim(0, max(d + v for d, v in zip(det, vlm)) * 1.12)
    a2.legend(fontsize=8, frameon=False, loc="upper right")
    _style(a2, "What one camera costs", "", "share of the GPU per camera (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


# ── cross-arm figures: one colour per arm on every figure ────────────────────
ARMS = [("B0_vllm_awq_clean", "vLLM, Qwen2.5-VL-7B", SERIES[1], "-"),
        ("C0_sglang_awq_clean", "SGLang, Qwen2.5-VL-7B", SERIES[0], "-"),
        ("Q3_vllm_awq_clean", "vLLM, Qwen3-VL-8B", SERIES[3], "-"),
        ("B_vllm_awq", "vLLM defaults (caches on)", WARN, "--")]


def _sweep(root, name):
    f = Path(root) / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def plot_e1_arms(root, out):
    """Single-stream decode against each arm's own bandwidth ceiling."""
    import yaml
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    xs, labels = [], []
    for i, (arm, lab, c, _) in enumerate(ARMS):
        d = _sweep(root, f"e1-roofline-{arm}")
        if not d:
            continue
        r = dict(d["rows"][0])
        # The ceiling comes from the arm's EXACT decode-weight bytes. Two stored
        # sweeps (SGLang, vLLM defaults) predate the exact basis and carry the
        # 5.807 GB estimate -- plotted as stored, SGLang read 94.7% instead of 90.8%.
        spec = yaml.safe_load(Path(f"arms/{arm}.yaml").read_text())
        gb = float(spec.get("decode_weight_bytes") or spec["weight_bytes"]) / 1e9
        r["decode_weight_GB"] = gb
        r["roofline_tok_s"] = 936.0 / gb
        r["pct_of_roofline"] = 100 * r["decode_tok_s"] / r["roofline_tok_s"]
        ax.bar(i, r["roofline_tok_s"], width=0.62, color="none", edgecolor=c, lw=1.4,
               ls="--")
        ax.bar(i, r["decode_tok_s"], width=0.62, color=c)
        ax.annotate(f"{r['pct_of_roofline']:.1f}%", (i, r["decode_tok_s"]),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    fontsize=9, color="white", fontweight="bold")
        ax.annotate(f"ceiling {r['roofline_tok_s']:.0f}", (i, r["roofline_tok_s"]),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7, color=FG)
        xs.append(i)
        labels.append(f"{lab}\n{r['decode_weight_GB'] or r['weight_GB']:.2f} GB/token")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 200)
    _style(ax, "Batch-1 decode: measured (solid) vs memory-bandwidth ceiling (dashed)", "",
           "tokens/s")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e2_arms(root, out):
    """TTFT against vision tokens, per arm, with each arm's linear fit."""
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    # ARMS[:3]: the defaults arm is left out on purpose -- its 115 ms/1k-token slope
    # is the synthetic-image cache-hit artefact documented in corrections.md
    for arm, lab, c, ls in ARMS[:3]:
        d = _sweep(root, f"e2-ttft-{arm}")
        if not d:
            continue
        rows = [r for r in d["rows"] if r.get("vision_tokens")]
        x = [r["vision_tokens"] for r in rows]
        ax.plot(x, [r["ttft_p50_ms"] for r in rows], "o", color=c, ms=6, mec="white", mew=1)
        fit = d["meta"].get("fit")
        if fit:
            xs = [0, max(x)]
            ax.plot(xs, [fit["intercept_ms"] + fit["slope_ms_per_token"] * v for v in xs], ls,
                    color=c, lw=1.6,
                    label=f"{lab}: {fit['intercept_ms']:.0f} ms + "
                          f"{fit['slope_ms_per_token']*1000:.0f} ms/1k tok")
    ax.set_xlim(0); ax.set_ylim(0)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(ax, "What a vision token costs, per stack and model", "vision tokens",
           "TTFT p50 (ms)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e0_arms(root, out):
    """Raw throughput vs throughput that meets the SLO, per arm."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.1))
    top = 0
    for arm, lab, c, ls in ARMS[:3]:
        d = _sweep(root, f"e0-saturation-{arm}")
        if not d:
            continue
        rows = d["rows"]
        # failure share per rate, from the run records: a rate where requests
        # failed is not a measurement (latency and goodput cover survivors only)
        for r in rows:
            f = Path("results") / f"e0-saturation-{arm}-qps{r['offered_qps']:g}" / "summary.json"
            sm = json.loads(f.read_text()) if f.exists() else {}
            n = sm.get("n_requests") or 0
            r["_failed"] = (1 - sm["n_ok"] / n) if n else 0.0
        good = [r for r in rows if r["_failed"] <= 0.01]
        bad = [r for r in rows if r["_failed"] > 0.01]
        for ax_, key in ((a1, "achieved_qps"), (a2, "goodput_req_s")):
            ax_.plot([r["offered_qps"] for r in good], [r[key] or 0 for r in good], "o" + ls,
                     color=c, lw=2, label=lab)
            if bad:
                ax_.plot([r["offered_qps"] for r in bad], [r[key] or 0 for r in bad], "o",
                         mfc="none", mec=c, mew=1.5, ms=8)
                for r in bad:
                    ax_.annotate(f"{100*r['_failed']:.0f}% failed", (r["offered_qps"], r[key] or 0),
                                 textcoords="offset points", xytext=(6, 4), fontsize=7, color=FG)
        top = max(top, max(r["offered_qps"] for r in rows))
    a1.plot([0, top], [0, top], ":", color=WARN, lw=1, label="offered")
    a1.plot([], [], "o", mfc="none", mec=FG, label="rate with failed requests (not a measurement)")
    a1.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(a1, "What the server completes", "offered load (req/s)", "completed (req/s)")
    a2.legend(fontsize=7.5, frameon=False, loc="upper right")
    _style(a2, "What it completes inside the SLO (TTFT <= 1 s)", "offered load (req/s)",
           "goodput (req/s)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_stacks_gap(root, out):
    """vLLM - SGLang accuracy gap by budget, default kernels vs both on SDPA."""
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    pairs = [("B0_vllm_awq_clean", "C0_sglang_awq_clean", "each stack's default vision attention",
              SERIES[1]),
             ("B0S_vllm_sdpa", "C0S_sglang_sdpa", "both pinned to SDPA", SERIES[0])]
    for a, b, lab, c in pairs:
        da, db = _sweep(root, f"e3-budget-{a}"), _sweep(root, f"e3-budget-{b}")
        if not (da and db):
            continue
        ra = {r["max_pixels"]: r for r in da["rows"]}
        rb = {r["max_pixels"]: r for r in db["rows"]}
        common = sorted(set(ra) & set(rb))
        x = [ra[k]["prompt_tokens"] for k in common]
        y = [100 * (ra[k]["accuracy"] - rb[k]["accuracy"]) for k in common]
        ax.plot(x, y, "o-", color=c, lw=2, label=lab)
    ax.axhline(0, color=FG, lw=0.7)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    _style(ax, "vLLM minus SGLang, DocVQA (ANLS points)", "prompt tokens",
           "accuracy gap (points)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e7_capacity(root, out, fresh_ms=2000):
    """Tail latency against offered rate for both arms and both sweep sizes."""
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    for which, c in (("full", SERIES[1]), ("roi", SERIES[0])):
        for suffix, mk, ls, n in (("", "o", "-", 160), ("-n80", "s", ":", 80)):
            d = _sweep(root, f"e7-capacity-{which}-V_vllm_video{suffix}")
            if not d:
                continue
            rows = d["rows"]
            ax.plot([r["achieved_qps"] for r in rows], [r["ttft_p99_ms"] / 1000 for r in rows],
                    mk + ls, color=c, lw=1.8 if n == 160 else 1.1, ms=6 if n == 160 else 4,
                    label=f"{'full frame' if which == 'full' else 'ROI crop'} (n={n}/rate)")
    ax.axhline(fresh_ms / 1000, color=ACCENT, lw=1, ls="--")
    ax.annotate("2 s freshness budget", (0.95, fresh_ms / 1000), textcoords="offset points",
                xytext=(0, 4), fontsize=7, color=FG)
    ax.set_yscale("log")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    _style(ax, "E7 capacity: where answers stop being fresh", "achieved windows/s",
           "TTFT p99 (s, log)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


# ── E7b · the live cascade ───────────────────────────────────────────────────
E7B_CONFIGS = [("dense-full-none", "VLM alone, every window", WARN, "o"),
               ("motion-full-pytorch", "motion gate, full frame, PyTorch YOLO", SERIES[1], "o"),
               ("motion-full-trt", "motion gate, full frame, TensorRT YOLO", SERIES[1], "s"),
               ("person-motion-roi-pytorch", "person-motion, ROI crop, PyTorch YOLO", SERIES[0], "o"),
               ("person-motion-roi-trt", "person-motion, ROI crop, TensorRT YOLO", SERIES[0], "s")]


def e7b_rows(tag, roots=("results/e7b", "results/e7b-ext")):
    """Every live run for one configuration, main sweep and extension merged."""
    rows = {}
    for r in roots:
        f = Path(r) / f"{tag}.json"
        if f.exists():
            for row in json.loads(f.read_text())["rows"]:
                rows[row["cams"]] = row
    return [rows[k] for k in sorted(rows)]


def e7b_max_supported(rows):
    """Highest N such that it and every tested smaller N passed."""
    best = 0
    for r in rows:
        if not r["supported"]:
            break
        best = r["cams"]
    return best


def plot_e7b_live(out):
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.2, 4.3))
    for tag, lab, c, mk in E7B_CONFIGS:
        rows = e7b_rows(tag)
        if not rows:
            continue
        ls = "--" if mk == "s" else "-"
        n = [r["cams"] for r in rows]
        a1.plot(n, [100 * (r["fresh_fraction"] or 0) for r in rows], mk + ls, color=c, lw=1.8,
                ms=6, label=lab)
        if tag != "dense-full-none":
            a2.plot(n, [1000 * (r["det_latency_p99_s"] or 0) for r in rows], mk + ls, color=c,
                    lw=1.8, ms=6, label=lab)
        # points, not lines: runs come from two server sessions, and vLLM's footprint
        # carries over within a session, so adjacent N are not comparable states
        for r in rows:
            a3.plot(r["cams"], (r["gpu_mem_max_mib"] or 0) / 1024, mk, color=c, ms=7,
                    mfc=c if r["supported"] else "none", mec=c, mew=1.5)
    a1.axhline(95, color=ACCENT, lw=1, ls=":")
    a1.annotate("95% fresh (< 2 s) required", (1, 95), textcoords="offset points",
                xytext=(0, -12), fontsize=7, color=FG)
    a1.set_ylim(-3, 104)
    a1.legend(fontsize=6.8, frameon=False, loc="lower left")
    _style(a1, "Answers under 2 s old", "cameras", "fresh answers (%)")
    a2.axhline(1000, color=ACCENT, lw=1, ls=":")
    a2.annotate("frames older than 1 s are dropped", (1, 1000), textcoords="offset points",
                xytext=(0, 4), fontsize=7, color=FG)
    a2.set_yscale("log")
    _style(a2, "Detector latency p99, sharing the GPU", "cameras", "capture to detections (ms, log)")
    a3.axhline(24.0, color=ACCENT, lw=1, ls=":")
    a3.annotate("24 GiB: the card", (1, 24.0), textcoords="offset points", xytext=(0, 4),
                fontsize=7, color=FG)
    a3.set_ylim(15, 25)
    a3.plot([], [], "o", color=FG, label="met the freshness bar")
    a3.plot([], [], "o", mfc="none", mec=FG, label="overloaded")
    a3.legend(fontsize=7.5, frameon=False, loc="lower right")
    _style(a3, "Peak GPU memory per run, both models", "cameras", "GiB")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def plot_e7b_model(out, modelled):
    """Measured supported cameras against the E7 time-sharing model."""
    tags = [t for t in E7B_CONFIGS if e7b_rows(t[0])]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    w = 0.38
    for i, (tag, lab, c, _) in enumerate(tags):
        meas = e7b_max_supported(e7b_rows(tag))
        mod = modelled.get(tag)
        ax.bar(i + w / 2, meas, width=w, color=c)
        ax.annotate(f"{meas}", (i + w / 2, meas), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=9, color=FG)
        if mod:
            ax.bar(i - w / 2, mod, width=w, color="none", edgecolor=c, lw=1.4, hatch="//")
            ax.annotate(f"{mod:.1f}", (i - w / 2, mod), textcoords="offset points",
                        xytext=(0, 3), ha="center", fontsize=8, color=FG)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([t[1].replace(", ", "\n", 1) for t in tags], fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="none", edgecolor=FG, hatch="//",
                             label="E7 model (detector and VLM measured separately)"),
                       Patch(facecolor=WARN, label="E7b measured live (both on the card)")],
              fontsize=8, frameon=False, loc="upper left")
    _style(ax, "Cameras one RTX 3090 carries: modelled vs measured", "", "cameras")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


# ── E7c · the DeepStream cascade ─────────────────────────────────────────────
def plot_e7c(out):
    """E7b's Python detector vs E7c's DeepStream pipeline, same cameras, same bar."""
    pairs = [("motion-full-trt", "track-motion-full-deepstream", "full frame", SERIES[1]),
             ("person-motion-roi-trt", "person-track-motion-roi-deepstream", "ROI crop", SERIES[0])]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.2, 4.3))
    for b_tag, c_tag, lab, col in pairs:
        rb = e7b_rows(b_tag)
        f = Path("results/e7c") / f"{c_tag}.json"
        rc = json.loads(f.read_text())["rows"] if f.exists() else []
        for rows, ls, mk, who in ((rb, "--", "s", "E7b TensorRT"),
                                  (rc, "-", "o", "E7c DeepStream")):
            if not rows:
                continue
            n = [r["cams"] for r in rows]
            a1.plot(n, [100 * (r["fresh_fraction"] or 0) for r in rows], mk + ls, color=col, lw=1.8,
                    ms=6, label=f"{lab} — {who}")
            a2.plot(n, [1000 * (r["det_latency_p99_s"] or 0) for r in rows], mk + ls, color=col,
                    lw=1.8, ms=6)
    a1.axhline(95, color=ACCENT, lw=1, ls=":")
    a1.set_ylim(-3, 104); a1.set_xlim(3.5, 14.5)
    a1.legend(fontsize=6.5, frameon=False, loc="center left")
    _style(a1, "Answers under 2 s old", "cameras", "fresh answers (%)")
    a2.set_yscale("log")
    a2.axhline(1000, color=ACCENT, lw=1, ls=":")
    _style(a2, "Detection latency p99", "cameras", "capture to detections (ms, log)")

    g = Path("results/e7c/gates.json")
    e7 = Path("results/e7/gates.json")
    if g.exists() and e7.exists():
        tr = {(r["gate"], r["bin"]): r for r in json.loads(g.read_text())["rows"]}
        bx = json.loads(e7.read_text())
        pts = [("motion", "track-motion", SERIES[1]), ("person-motion", "person-track-motion", SERIES[0])]
        for box_g, trk_g, col in pts:
            x0 = 100 * bx["calls"][f"1280|{box_g}|all"]["rate"]
            y0 = 100 * bx["recall"][f"1280|{box_g}|all"]["rate"]
            x1 = 100 * tr[(trk_g, "all")]["call_rate"]
            y1 = 100 * tr[(trk_g, "all")]["recall"]
            a3.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
            a3.plot([x0], [y0], "s", color=col, ms=7, mfc="none", mew=1.5)
            a3.plot([x1], [y1], "o", color=col, ms=7)
            a3.annotate(trk_g, (x1, y1), textcoords="offset points", xytext=(8, 4),
                        ha="left", fontsize=7.5, color=FG)
        a3.plot([], [], "s", mfc="none", mec=FG, label="E7 box-matching gate")
        a3.plot([], [], "o", color=FG, label="E7c tracker gate")
        a3.legend(fontsize=7.5, frameon=False, loc="lower right")
    _style(a3, "Offline: what the gate sends and keeps", "windows sent (%)",
           "annotated activities covered (%)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)


E7D_MODELS = [  # (arm id, label, generation index, size in B)
    ("V_vllm_video", "Qwen2.5-VL-7B", 0, 7), ("E_q25_3b", "Qwen2.5-VL-3B", 0, 3),
    ("E_q3vl_8b", "Qwen3-VL-8B", 1, 8), ("E_q3vl_4b", "Qwen3-VL-4B", 1, 4),
    ("E_q3vl_2b", "Qwen3-VL-2B", 1, 2), ("E_q35_9b", "Qwen3.5-9B", 2, 9),
    ("E_q35_4b", "Qwen3.5-4B", 2, 4), ("E_q35_2b", "Qwen3.5-2B", 2, 2)]
E7D_SHORT = {"A": "in/out of vehicle", "B": "vehicle moves", "C": "doorway", "D": "object handling",
             "E": "phone", "F": "conversation", "G": "sit / stand", "H": "bike, laptop, reading"}
E7D_GEN = ["Qwen2.5-VL (Jan 2025)", "Qwen3-VL (Oct 2025)", "Qwen3.5 (Feb 2026)"]


def e7d_cameras(arm_id, view, with_limiter=False):
    """Highest supported camera count (None if no live run), its peak memory, and
    what failed one step later: 'det' if only detection latency broke the bar.
    Qwen3-VL-8B's warmed-up rerun is used where it exists (its first sweep failed
    on start-up detection latency; both are on the E7d page)."""
    tag = {"full": "track-motion-full-deepstream",
           "roi": "person-track-motion-roi-deepstream"}[view]
    d = Path("results/e7c") if arm_id == "V_vllm_video" else Path("results/e7d") / arm_id
    if (d / "rerun" / f"{tag}.json").exists():
        d = d / "rerun"
    f = d / f"{tag}.json"
    if not f.exists():
        return (None, None, None) if with_limiter else (None, None)
    rows = json.loads(f.read_text())["rows"]
    ok = [r for r in rows if r["supported"]]
    best = max(ok, key=lambda r: r["cams"]) if ok else None
    cams = best["cams"] if best else 0
    nxt = min((r for r in rows if not r["supported"] and r["cams"] > cams),
              key=lambda r: r["cams"], default=None)
    lim = None
    if nxt:
        det_bad = (nxt["det_latency_p99_s"] or 0) >= 1.0
        lim = "det" if det_bad and (nxt["fresh_fraction"] or 0) >= 0.9 else "vlm"
    mem = best.get("gpu_mem_max_mib") if best else None
    return (cams, mem, lim) if with_limiter else (cams, mem)


def plot_e7d(out_frontier, out_groups, group_names):
    comp = json.loads(Path("results/e7d/compare.json").read_text())["rows"]
    lift = {(r["arm_id"], r["view"]): r for r in comp}

    def skill_ci(arm_id, view):
        f = (Path("results/e7/cascades.json") if arm_id == "V_vllm_video"
             else Path("results/e7d") / arm_id / "cascades.json")
        if not f.exists():
            return None
        for r in json.loads(f.read_text())["rows"]:
            if (r["size"] == 1280 and r["gate"] == "dense" and r["arm"] == view
                    and r["bin"] == "all" and r.get("lift_ci95")):
                return r["lift_ci95"]
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), sharey=True)
    for ax, view, title, gate in ((axes[0], "full", "Full frame", "track-motion gate"),
                                  (axes[1], "roi", "ROI crop", "person-track-motion gate")):
        base_c, _ = e7d_cameras("V_vllm_video", view)
        ax.axvspan(18.5, 20.5, color=WARN, alpha=0.08, lw=0)
        ax.text(19.5, 0.97, "detector\nceiling", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=7, color=WARN)
        if base_c:
            ax.axvline(base_c, color=WARN, lw=0.8, ls=":")
        ax.axhline(0, color=WARN, lw=0.8)
        for aid, lab, gen, size in E7D_MODELS:
            r = lift.get((aid, view))
            cams, _, lim = e7d_cameras(aid, view, with_limiter=True)
            if r is None or cams is None:
                continue
            y = 100 * r["lift"]
            col = SERIES[gen]
            unusable = (r["compliance"] or 0) < 0.5 or r["false_alarms_per_hour"] > 5000
            ci = skill_ci(aid, view)
            x = cams + (gen - 1) * 0.12          # models tied on cameras stay separable
            if ci:
                ax.plot([x, x], [100 * ci[0], 100 * ci[1]], color=col, lw=1.2, alpha=0.45)
            ax.plot([x], [y], "o", ms=5 + size, color=col,
                    mfc="none" if unusable else col, mew=1.8, zorder=3)
            note = " (format fails)" if (r["compliance"] or 0) < 0.5 else (
                " (ticks every box)" if r["false_alarms_per_hour"] > 5000 else "")
            if lim == "det":
                note += " · detector-limited"
            ax.annotate(lab.split("-")[-1] + note, (x, y), textcoords="offset points",
                        xytext=(7 + size / 2, -3), fontsize=7.5, color=FG)
        _style(ax, f"{title} — {gate}", "cameras per RTX 3090 (live, DeepStream + vLLM)",
               "recognition above chance (points)" if view == "full" else "")
    for g, name in enumerate(E7D_GEN):
        axes[0].plot([], [], "o", color=SERIES[g], label=name)
    axes[0].plot([], [], "o", mfc="none", mec=FG, label="unusable answers")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(out_frontier, dpi=160)
    plt.close(fig)

    # which activities each model sees: lift per group, full frame
    models = [(aid, lab) for aid, lab, *_ in E7D_MODELS if (aid, "full") in lift]
    mute = {aid for aid, _ in models if (lift[(aid, "full")]["compliance"] or 0) < 0.5}
    groups = sorted({g for aid, _ in models for g in lift[(aid, "full")]["per_group"]})
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    m = np.array([[100 * lift[(aid, "full")]["per_group"].get(g, {}).get("lift", np.nan)
                   for g in groups] if aid not in mute else [np.nan] * len(groups)
                  for aid, _ in models])
    cmap = LinearSegmentedColormap.from_list("div", [SERIES[0], "#f2f2f2", SERIES[1]])
    fig, ax = plt.subplots(figsize=(11.5, 0.42 * len(models) + 1.9))
    ax.imshow(m, cmap=cmap, norm=TwoSlopeNorm(0, -60, 60), aspect="auto")
    for i in range(len(models)):
        for j in range(len(groups)):
            if not np.isnan(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(m[i, j]) > 40 else FG)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color=WARN)
    n = {g: lift[(models[0][0], "full")]["per_group"].get(g, {}).get("n", 0) for g in groups}
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}. {E7D_SHORT.get(g, g)}\n(n={n[g]})" for g in groups], fontsize=7.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([lab + (" (no letters)" if aid in mute else "") for aid, lab in models],
                       fontsize=8)
    ax.set_title("Recognition above chance by activity group, full frame (points; blue = better "
                 "than chance)", loc="left", fontsize=10, color=FG, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_groups, dpi=160)
    plt.close(fig)

def comparisons(root="results/sweeps", out_dir="docs/img"):
    """Build the cross-sweep figures the comparison pages need."""
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for fn, name in ((plot_headline, "headline.png"), (plot_e4, "e4-video.png"),
                     (plot_corrections, "corrections.png")):
        fn(root, out_dir / name)
        print(f"  wrote {out_dir/name}")

    e7 = Path("results/e7")
    if (e7 / "gates.json").exists():
        plot_e7_gates(e7 / "gates.json", out_dir / "e7-gates.png")
        plot_e7_detector(e7 / "gates.json", out_dir / "e7-detector.png")
        print(f"  wrote {out_dir/'e7-gates.png'}, {out_dir/'e7-detector.png'}")
    if (e7 / "cascades.json").exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from e7_vlm import GROUPS
        plot_e7_groups(e7 / "cascades.json", out_dir / "e7-groups.png",
                       {g: d[0] for g, d in GROUPS.items()})
        plot_e7_cameras(e7 / "cascades.json", out_dir / "e7-cameras.png")
        print(f"  wrote {out_dir/'e7-groups.png'}, {out_dir/'e7-cameras.png'}")

    for fn, name in ((plot_e1_arms, "e1-arms.png"), (plot_e2_arms, "e2-arms.png"),
                     (plot_e0_arms, "e0-arms.png"), (plot_stacks_gap, "stacks-gap.png"),
                     (plot_e7_capacity, "e7-capacity.png")):
        fn(root, out_dir / name)
        print(f"  wrote {out_dir/name}")

    if Path("results/e7b").exists():
        plot_e7b_live(out_dir / "e7b-live.png")
        cas = Path("results/e7/cascades.json")
        modelled = {}
        if cas.exists():
            rows = {(r["gate"], r["arm"]): r for r in json.loads(cas.read_text())["rows"]
                    if r["bin"] == "all" and r["size"] == 1280}
            modelled = {"dense-full-none": rows[("dense", "full")]["cameras_per_gpu"],
                        "motion-full-pytorch": rows[("motion", "full")]["cameras_per_gpu"],
                        "person-motion-roi-pytorch": rows[("person-motion", "roi")]["cameras_per_gpu"]}
        plot_e7b_model(out_dir / "e7b-model.png", modelled)
        print(f"  wrote {out_dir/'e7b-live.png'}, {out_dir/'e7b-model.png'}")

    if Path("results/e7c").exists():
        plot_e7c(out_dir / "e7c.png")
        print(f"  wrote {out_dir/'e7c.png'}")

    if Path("results/e7d/compare.json").exists():
        sys.path.insert(0, str(Path(__file__).parent))
        from e7_vlm import GROUPS
        plot_e7d(out_dir / "e7d-frontier.png", out_dir / "e7d-groups.png",
                 {g: d[0] for g, d in GROUPS.items()})
        print(f"  wrote {out_dir/'e7d-frontier.png'}, {out_dir/'e7d-groups.png'}")

    tasks = [root / f"e3-{t}.json" for t in TASKS]
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
