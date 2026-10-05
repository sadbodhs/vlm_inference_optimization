#!/usr/bin/env python3
"""E9 concept figure: what each input sends, and where the tokens are cut.

Three rows, the same 2-second window (8 frames = 4 merged frame pairs):
  1. 8 frames as video, vLLM EVS: the encoder processes all 4 pairs, then the
     least-changed tokens are dropped before the LLM.
  2. 8 frames, pruned before the encoder (E9): unchanged 28 px units are dropped
     from the pixels, so the encoder only processes what the LLM will receive.
  3. 2 frames as video (E8): one pair, nothing to prune.
Box widths are schematic; the numbers on them are measured (E8, E9).

    python3 tools/e9_concept_fig.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

FG, MUTED, KEEP, DROP, ENC, LLM = "#1a1a1a", "#888888", "#2a64b4", "#e6e6e6", "#c1440e", "#1f8a52"
rng = np.random.default_rng(3)
GH, GW = 6, 10                                  # a coarse stand-in for the 18 x 32 unit grid


def grid(ax, x, y, w, h, keep, label=None):
    cw, ch = w / GW, h / GH
    for r in range(GH):
        for c in range(GW):
            ax.add_patch(Rectangle((x + c * cw, y + (GH - 1 - r) * ch), cw * 0.92, ch * 0.92,
                                   color=KEEP if keep[r, c] else DROP, lw=0))
    if label:
        ax.text(x + w / 2, y - 0.18, label, ha="center", va="top", fontsize=7, color=FG)


def box(ax, x, y, w, h, color, text, sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                color=color, alpha=0.9, lw=0))
    ax.text(x + w / 2, y + h / 2 + (0.1 if sub else 0), text, ha="center", va="center",
            fontsize=8, color="white", weight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=7, color="white")


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))


# one moving person: a small blob that shifts right pair by pair
moving = []
for k in range(4):
    m = np.zeros((GH, GW), bool)
    m[3:5, 2 + 2 * k:4 + 2 * k] = True
    moving.append(m)
full = np.ones((GH, GW), bool)

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.6); ax.axis("off")
rows = [
    (4.6, "8 frames as video,\nvLLM EVS (rate 0.5)",
     [full] * 4, [full] * 4, "encoder: all 4 pairs", "240 ms (8 frames)",
     [full] + [moving[k] | (rng.random((GH, GW)) < 0.25) for k in (1, 2, 3)], "LLM: 1,293 tokens",
     "pruned AFTER the encoder: the encoder already paid for everything", "6 cameras"),
    (2.5, "8 frames, pruned\nbefore the encoder\n(E9, ours)",
     [full] + moving[1:], [full] + moving[1:], "encoder: kept units only", "86 ms (rate 0.675)",
     [full] + moving[1:], "LLM: 902–1,293 tokens",
     "pruned BEFORE the encoder: unchanged units are never encoded", "9–10 cameras"),
    (0.4, "2 frames as video\n(E8)",
     [full], [full], "encoder: 1 pair", "",
     [full], "LLM: 735 tokens", "one frame pair: nothing to prune", "15 cameras"),
]
for y, name, sent, enc_in, enc_t, enc_s, out, llm_t, note, cams in rows:
    ax.text(0.05, y + 0.62, name, fontsize=8.5, color=FG, va="center", weight="bold")
    x = 2.0
    for k, m in enumerate(sent):
        grid(ax, x + k * 0.95, y + 0.2, 0.85, 0.85, m, f"pair {k + 1}" if len(sent) > 1 else "frames 18 + 48")
    xe = 2.0 + 4 * 0.95 + 0.25
    arrow(ax, xe - 0.2, xe, y + 0.62)
    encw = 1.9 if enc_t.startswith("encoder: all") else (1.15 if "kept" in enc_t else 0.75)
    box(ax, xe, y + 0.25, encw, 0.75, ENC, enc_t.split(": ")[0], enc_t.split(": ")[1] if not enc_s else enc_s)
    xo = xe + 2.1
    arrow(ax, xe + encw + 0.03, xo - 0.05, y + 0.62)
    for k, m in enumerate(out):
        grid(ax, xo + k * 0.62, y + 0.32, 0.55, 0.55, m)
    xl = xo + 4 * 0.62 + 0.15
    arrow(ax, xl - 0.12, xl, y + 0.62)
    box(ax, xl, y + 0.25, 1.45, 0.75, LLM, llm_t.split(": ")[0], llm_t.split(": ")[1])
    ax.text(xl + 1.6, y + 0.62, cams, fontsize=9, color=FG, va="center", weight="bold")
    ax.text(2.0, y - 0.12, note, fontsize=7.5, color=MUTED, va="top", style="italic")

ax.text(2.0, 6.35, "what the vision encoder processes (blue = a 28 px unit = one token per frame pair)",
        fontsize=8, color=FG)
ax.text(8.3, 6.35, "tokens reaching the LLM", fontsize=8, color=FG)
ax.text(12.2, 6.35, "live cameras\n(Qwen2.5-VL-7B)", fontsize=8, color=FG, ha="center")
fig.tight_layout()
fig.savefig("docs/img/e9-concept.png", dpi=160)
print("wrote docs/img/e9-concept.png")
