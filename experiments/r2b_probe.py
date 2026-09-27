#!/usr/bin/env python3
"""R2b (RECIPE.md): a frozen-encoder probe -- does a hand crop hold more information
about the assembly step than the full frame, for a trained head?

  prep      crops for every sample of the 20-worker subset (Kinect body tracking,
            label-free): tight (1 shoulder width around each wrist), wide (3), bench
  features  Qwen3-VL-8B's vision tower, frozen, loaded alone; each image's merged
            tokens mean-pooled, the two frames averaged -> one vector per sample/view
  probe     multinomial logistic regression, L2 from an inner split, 5-fold CV
            grouped by worker; paired worker bootstrap between views

    docker/run_harness.sh python3 experiments/r2b_probe.py prep
    scripts/r2b.sh features          # GPU, in the vLLM image
    docker/run_harness.sh python3 experiments/r2b_probe.py probe   (or on the Mac)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path("data/ha4m")
FEAT = ROOT / "r2b_features"
OUT = Path("results/r2b")
VIEWS = ["full", "tight", "wide", "bench"]
MAX_PX, MIN_PX = 451_584, 65_536
PART = set(range(1, 9)) | {10, 11}


# ── prep ─────────────────────────────────────────────────────────────────────
def cmd_prep(args):
    import r2_prep as P
    man = [s for s in json.loads((ROOT / "manifest.json").read_text()) if s["complete"]]
    subs = sorted({s["subject"] for s in man})
    first = {sub: next(s for s in man if s["subject"] == sub) for sub in subs}
    refs = {1: P.thumb(first[subs[0]]["rec"], first[subs[0]]["frames"][0]),
            2: P.thumb(first[subs[-1]]["rec"], first[subs[-1]]["frames"][0])}
    setup = {sub: P.setup_of(first[sub]["rec"], first[sub]["frames"][0], refs) for sub in subs}

    def box(parts, width):
        """P.hand_box with the square scaled by `width` shoulder widths."""
        sq = []
        for p in parts:
            if not p or not p["wrists"]:
                continue
            import math
            sw = (math.dist(*p["shoulders"]) if len(p["shoulders"]) == 2 else 0) or 160.0
            s = max(96.0, sw) * width
            sq += [(x - s / 2, y - s / 2, x + s / 2, y + s / 2) for x, y in p["wrists"]]
        if not sq:
            return None
        x1, y1 = max(0, min(b[0] for b in sq)), max(0, min(b[1] for b in sq))
        x2, y2 = min(P.FW, max(b[2] for b in sq)), min(P.FH, max(b[3] for b in sq))
        return [int(x1), int(y1), int(x2), int(y2)] if x2 - x1 > 32 and y2 - y1 > 32 else None

    bench = {}
    for k in (1, 2):
        xs, ys = [], []
        for s in man:
            if setup[s["subject"]] == k:
                for f in s["frames"]:
                    for x, y in (P.kinect(s["rec"], f) or {}).get("wrists", []):
                        xs.append(x); ys.append(y)
        xs.sort(); ys.sort()
        q = lambda v, p: v[int(p * (len(v) - 1))]
        cx, cy = (q(xs, .05) + q(xs, .95)) / 2, (q(ys, .05) + q(ys, .95)) / 2
        w, h = (q(xs, .95) - q(xs, .05)) * 1.2, (q(ys, .95) - q(ys, .05)) * 1.2
        bench[k] = [int(max(0, cx - w / 2)), int(max(0, cy - h / 2)),
                    int(min(P.FW, cx + w / 2)), int(min(P.FH, cy + h / 2))]
    out, fb = [], collections.Counter()
    for s in man:
        kp = [P.kinect(s["rec"], f) for f in s["frames"]]
        t, w = box(kp, 1.0), box(kp, 3.0)
        fb["tight"] += t is None; fb["wide"] += w is None
        out.append({**s, "id": f"{s['rec']}|{s['step']}|{s['frames'][0]}", "setup": setup[s["subject"]],
                    "crop": {"full": None, "tight": t, "wide": w, "bench": bench[setup[s["subject"]]]}})
    (ROOT / "r2b_samples.json").write_text(json.dumps(out))
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1]) / (P.FW * P.FH)
    facts = {"samples": len(out), "workers": len(subs),
             "per_setup": dict(collections.Counter(setup[s["subject"]] for s in man)),
             "workers_per_setup": dict(collections.Counter(setup.values())),
             "fallback_to_full": dict(fb), "bench": bench,
             "crop_area_frac_median": {v: st.median(area(s["crop"][v]) for s in out if s["crop"][v])
                                       for v in ("tight", "wide", "bench")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "facts.json").write_text(json.dumps(facts, indent=1))
    print(json.dumps(facts, indent=1))


# ── features ─────────────────────────────────────────────────────────────────
def load_tower(snap: str):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    cfg = AutoConfig.from_pretrained(snap)
    vm = Qwen3VLVisionModel._from_config(cfg.vision_config, torch_dtype=torch.bfloat16)
    sd = {}
    for f in sorted(glob.glob(f"{snap}/*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                for pre in ("model.visual.", "visual."):
                    if k.startswith(pre):
                        sd[k[len(pre):]] = fh.get_tensor(k)
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"vision tower: {len(missing)} missing weights, e.g. {missing[:3]}")
    print(f"vision tower loaded: {sum(p.numel() for p in vm.parameters())/1e6:.0f}M params, "
          f"{len(unexpected)} unexpected keys", flush=True)
    return vm.cuda().eval()


def cmd_features(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor
    snap = glob.glob(f"/root/.cache/huggingface/hub/models--{args.model.replace('/', '--')}/snapshots/*")[0]
    proc = AutoImageProcessor.from_pretrained(snap)
    vm = load_tower(snap)
    samples = json.loads((ROOT / "r2b_samples.json").read_text())
    if args.limit:
        samples = samples[:args.limit]
    FEAT.mkdir(parents=True, exist_ok=True)
    merge = vm.config.spatial_merge_size ** 2

    def encode(imgs):
        enc = proc(images=imgs, return_tensors="pt",
                   size={"shortest_edge": MIN_PX, "longest_edge": MAX_PX})
        pv = enc["pixel_values"].cuda().to(torch.bfloat16)
        thw = enc["image_grid_thw"].cuda()
        with torch.no_grad():
            o = vm(pv, grid_thw=thw)
        h = o[0] if isinstance(o, (tuple, list)) else getattr(o, "pooler_output", None)
        if h is None:
            h = o.last_hidden_state
        n = (thw.prod(-1) // merge).tolist()
        return [c.float().mean(0).cpu().numpy() for c in torch.split(h, n)]

    for view in VIEWS:
        vecs = []
        for i in range(0, len(samples), args.batch):
            chunk = samples[i:i + args.batch]
            imgs = []
            for s in chunk:
                for f in s["frames"]:
                    im = Image.open(ROOT / s["rec"] / "color" / f"{f:06d}.png").convert("RGB")
                    b = s["crop"][view]
                    imgs.append(im.crop(tuple(b)) if b else im)
            e = encode(imgs)
            vecs += [(e[2 * j] + e[2 * j + 1]) / 2 for j in range(len(chunk))]
            if i % (args.batch * 20) == 0:
                print(f"  {view}: {i}/{len(samples)}", flush=True)
        np.save(FEAT / f"{view}.npy", np.stack(vecs).astype("float32"))
        print(f"  {view}: {len(vecs)} x {vecs[0].shape[0]}", flush=True)
    (FEAT / "ids.json").write_text(json.dumps([s["id"] for s in samples]))


# ── probe ────────────────────────────────────────────────────────────────────
def fit_predict(Xtr, ytr, Xte, l2, n_cls=13, iters=200):
    import torch
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    W = torch.zeros(Xtr.shape[1], n_cls, requires_grad=True)
    b = torch.zeros(n_cls, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtr @ W + b, ytr) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (Xte @ W + b).argmax(1)


def cmd_probe(args):
    import numpy as np
    import torch
    samples = {s["id"]: s for s in json.loads((ROOT / "r2b_samples.json").read_text())}
    ids = json.loads((FEAT / "ids.json").read_text())
    S = [samples[i] for i in ids]
    y = torch.tensor([s["step"] for s in S])
    X = {v: torch.tensor(np.load(FEAT / f"{v}.npy")) for v in VIEWS}
    X["full+wide"] = torch.cat([X["full"], X["wide"]], 1)
    workers = sorted({s["subject"] for s in S})
    by_setup = collections.defaultdict(list)
    for w in workers:
        by_setup[next(s["setup"] for s in S if s["subject"] == w)].append(w)
    folds = [[] for _ in range(5)]
    for k, ws in by_setup.items():              # each fold gets workers from both setups
        for i, w in enumerate(ws):
            folds[i % 5].append(w)
    wid = [s["subject"] for s in S]
    pred = {}
    for v, Xv in X.items():
        p = torch.empty_like(y)
        for f in folds:
            te = torch.tensor([w in f for w in wid])
            tr_workers = [w for w in workers if w not in f]
            rng = random.Random(0)
            inner = set(rng.sample(tr_workers, max(1, len(tr_workers) // 5)))
            itr = torch.tensor([(w not in f) and (w not in inner) for w in wid])
            iva = torch.tensor([w in inner for w in wid])
            best = max((1e-4, 1e-3, 1e-2, 1e-1),
                       key=lambda l2: (fit_predict(Xv[itr], y[itr], Xv[iva], l2) == y[iva]).float().mean().item())
            p[te] = fit_predict(Xv[~te], y[~te], Xv[te], best)
        pred[v] = p
        print(f"  {v:10s} accuracy {100*(p == y).float().mean():5.1f}%", flush=True)

    def acc(v, mask=None):
        m = torch.ones_like(y, dtype=torch.bool) if mask is None else mask
        return (pred[v][m] == y[m]).float().mean().item()

    def macro_f1(v):
        f = []
        for c in range(13):
            tp = ((pred[v] == c) & (y == c)).sum().item()
            fp = ((pred[v] == c) & (y != c)).sum().item()
            fn = ((pred[v] != c) & (y == c)).sum().item()
            if tp + fp + fn:
                f.append(2 * tp / (2 * tp + fp + fn))
        return st.mean(f)

    def paired(a, b, boot=1000, seed=7):
        rng, d = random.Random(seed), []
        idx = {w: [i for i, x in enumerate(wid) if x == w] for w in workers}
        for _ in range(boot):
            ii = [i for w in (rng.choice(workers) for _ in workers) for i in idx[w]]
            t = torch.tensor(ii)
            d.append(((pred[a][t] == y[t]).float().mean() - (pred[b][t] == y[t]).float().mean()).item())
        d.sort()
        return acc(a) - acc(b), d[int(.025 * boot)], d[int(.975 * boot) - 1]

    part = torch.tensor([s["step"] in PART for s in S])
    whole = torch.tensor([s["step"] in (9, 12) for s in S])
    majority = collections.Counter(y.tolist()).most_common(1)[0][1] / len(y)
    rows = {v: {"accuracy": acc(v), "macro_f1": macro_f1(v), "part_steps": acc(v, part),
                "steps_9_12": acc(v, whole),
                "per_setup": {k: acc(v, torch.tensor([s["setup"] == k for s in S])) for k in (1, 2)},
                "vs_full": paired(v, "full") if v != "full" else None} for v in pred}
    best_single = max(("full", "tight", "wide", "bench"), key=lambda v: rows[v]["accuracy"])
    v = {"R2b.1": {"holds": rows["full"]["accuracy"] >= 0.40, "full": rows["full"]["accuracy"]},
         "R2b.2": {"holds": rows["wide"]["accuracy"] - rows["full"]["accuracy"] >= 0.03,
                   "wide_minus_full": rows["wide"]["vs_full"]},
         "R2b.3": {"holds": rows["wide"]["accuracy"] >= rows["tight"]["accuracy"],
                   "wide_minus_tight": rows["wide"]["accuracy"] - rows["tight"]["accuracy"]},
         "R2b.4": {"holds": rows["full+wide"]["accuracy"] - rows[best_single]["accuracy"] >= 0.02,
                   "best_single": best_single,
                   "gain": rows["full+wide"]["accuracy"] - rows[best_single]["accuracy"],
                   "vs_best_single": paired("full+wide", best_single)},
         "R2b.5": {"holds": (rows["wide"]["part_steps"] - rows["full"]["part_steps"]) >
                            (rows["wide"]["steps_9_12"] - rows["full"]["steps_9_12"]),
                   "gain_part": rows["wide"]["part_steps"] - rows["full"]["part_steps"],
                   "gain_9_12": rows["wide"]["steps_9_12"] - rows["full"]["steps_9_12"]}}
    res = {"n": len(S), "workers": len(workers), "folds": folds, "majority_class": majority,
           "views": rows, "predictions": v,
           "per_step": {vv: {c: acc(vv, y == c) for c in range(13)} for vv in pred}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"\nmajority-class baseline {100*majority:.1f}%")
    for k, r in rows.items():
        vs = "" if not r["vs_full"] else "  vs full %+.1f [%+.1f, %+.1f]" % tuple(100 * x for x in r["vs_full"])
        print(f"  {k:10s} acc {100*r['accuracy']:5.1f}  macro-F1 {100*r['macro_f1']:5.1f}  "
              f"part {100*r['part_steps']:5.1f}  9/12 {100*r['steps_9_12']:5.1f}{vs}")
    for k, x in v.items():
        print(f"  {k}: {'HELD' if x['holds'] else 'failed'}  " + json.dumps(
            {a: (round(b, 3) if isinstance(b, float) else b) for a, b in x.items() if a != "holds"},
            default=lambda o: [round(q, 3) for q in o]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "features", "probe"])
    ap.add_argument("--model", default="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"prep": cmd_prep, "features": cmd_features, "probe": cmd_probe}[args.mode](args)


if __name__ == "__main__":
    main()
