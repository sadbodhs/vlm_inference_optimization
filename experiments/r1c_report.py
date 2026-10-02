#!/usr/bin/env python3
"""R1c (RECIPE.md): score the precision pass against R1b's first pass; check R1c.1-R1c.6.

Scoring is R1b's (r1b_report.score / paired). V arms: a first-pass letter survives if the
verify answer says yes for it; a letter the verify answer does not mention is kept and
counted as unparsed. Tokens per fired window sum both passes.

    python3 experiments/r1c_report.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics as st
from pathlib import Path

import r1b_report as rb
from r1b_report import load, paired, ratio, score

YESNO = re.compile(r"\b([A-H])\s*[:.)\-]?\s*(yes|no)\b", re.I)


def verify(base: tuple, d: Path):
    """Apply a verify run to a first pass: (answers, tokens, keep stats)."""
    ans1, tok1, _ = base
    says, tok2 = {}, collections.Counter()
    for line in (d / "outputs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["ok"]:
            says[r["sample_id"].rsplit("|", 1)[0]] = {g.upper(): v.lower() == "yes"
                                                      for g, v in YESNO.findall(r["text"] or "")}
    for line in (d / "latency.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("prompt_tokens") is not None:
            tok2[r["sample_id"].rsplit("|", 1)[0]] += r["prompt_tokens"]
    ans, unparsed = {}, 0
    for w, a in ans1.items():
        s = says.get(w, {})
        unparsed += sum(g not in s for g in a)
        ans[w] = {g for g in a if s.get(g, True)}
    return ans, tok1, tok2, unparsed


def keep_rates(wins, ans1, ans2):
    kc = tc = kf = tf = 0
    for w in wins:
        t, a, b = set(w["truth"]), ans1.get(w["id"], set()), ans2.get(w["id"], set())
        tc += len(a & t); kc += len(b & t)
        tf += len(a - t); kf += len(b - t)
    return (kc / tc if tc else None), (kf / tf if tf else None)


def precision(wins, ans):
    named = sum(len(ans.get(w["id"], set())) for w in wins)
    right = sum(len(ans.get(w["id"], set()) & set(w["truth"])) for w in wins)
    return right / named if named else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1-root", default="results/r1b")
    ap.add_argument("--root", default="results/r1c")
    ap.add_argument("--out", default="results/r1c/report.json")
    ap.add_argument("--selection", default="results/e7/selection.json")
    args = ap.parse_args()
    rb.CLIPS[:] = sorted(c["clip"] for c in json.load(open(args.selection))["clips"])
    W = json.loads((Path(args.pass1_root) / "windows.json").read_text())
    wins, acts = W["windows"], W["activities"]
    p1, root = Path(args.pass1_root), Path(args.root)
    first = {(n, m): load(p1 / f"{n}-{m}") for n in ("F", "GO") for m in ("E_q3vl_4b", "E_q3vl_8b")}
    rows, res = {}, {"models": {}}

    def add(key, ans, tok_by_model, ref, ans1=None, unparsed=None):
        L = score(acts, wins, ans)
        row = {"lift": L[0], "recognised": L[1], "chance": L[2], "fa_per_hour": L[3],
               "precision": precision(wins, ans),
               "tokens_per_window": {m: st.mean(t[w["id"]] for w in wins) for m, t in tok_by_model.items()},
               "vs_F": paired(acts, wins, ans, ref) if ref is not None else None}
        if ans1 is not None:
            row["keep_correct"], row["keep_false"] = keep_rates(wins, ans1, ans)
            row["unparsed_letters"] = unparsed
            row["verified_windows_share"] = sum(bool(ans1.get(w["id"])) for w in wins) / len(wins)
        rows[key] = row

    for m in ("E_q3vl_4b", "E_q3vl_8b"):
        F = first[("F", m)]
        if not F:
            continue
        for n in ("F", "GO"):
            if first[(n, m)]:
                a, t, _ = first[(n, m)]
                add(f"{n}|{m}", a, {m: t}, None if n == "F" else F[0])
        for n in ("Fc", "GOc"):
            if (root / f"{n}-{m}" / "outputs.jsonl").exists():
                a, t, _ = load(root / f"{n}-{m}")
                add(f"{n}|{m}", a, {m: t}, F[0])
        for n, b in (("FV", "F"), ("GOV", "GO")):
            if (root / f"{n}-{m}" / "outputs.jsonl").exists() and first[(b, m)]:
                a, t1, t2, u = verify(first[(b, m)], root / f"{n}-{m}")
                tok = collections.Counter(t1); tok.update(t2)
                add(f"{n}|{m}", a, {m: tok}, F[0], first[(b, m)][0], u)
    if (root / "F4V8-E_q3vl_8b" / "outputs.jsonl").exists() and first[("F", "E_q3vl_4b")]:
        a, t1, t2, u = verify(first[("F", "E_q3vl_4b")], root / "F4V8-E_q3vl_8b")
        ref8 = first[("F", "E_q3vl_8b")][0] if first[("F", "E_q3vl_8b")] else None
        add("F4V8|cascade", a, {"E_q3vl_4b": t1, "E_q3vl_8b": t2}, ref8,
            first[("F", "E_q3vl_4b")][0], u)

    print(f"{'arm':22s} {'lift':>6s} {'FA/h':>6s} {'prec':>5s} {'keep ok/false':>14s} "
          f"{'tok/win':>16s} {'vs F [95%]':>22s}")
    for k, r in rows.items():
        kp = "" if "keep_correct" not in r else f"{r['keep_correct']:.2f}/{r['keep_false']:.2f}"
        tk = "+".join(f"{v:.0f}" for v in r["tokens_per_window"].values())
        pv = r["vs_F"]
        vs = "" if not pv else f"{100*pv[0]:+5.1f} [{100*pv[1]:+5.1f},{100*pv[2]:+5.1f}]"
        print(f"{k:22s} {100*r['lift']:+6.1f} {r['fa_per_hour']:6.0f} {r['precision'] or 0:5.2f} "
              f"{kp:>14s} {tk:>16s} {vs:>22s}")
    res["rows"] = rows

    g = rows.get
    tok = lambda k, m: g(k)["tokens_per_window"][m]
    v, m4, m8 = {}, "E_q3vl_4b", "E_q3vl_8b"
    F4, F8 = g(f"F|{m4}"), g(f"F|{m8}")
    if g(f"FV|{m4}"):
        x = g(f"FV|{m4}")
        v["R1c.1"] = {"holds": x["fa_per_hour"] <= 0.5 * F4["fa_per_hour"] and x["lift"] >= F4["lift"] - 0.02,
                      "fa_ratio": ratio(x["fa_per_hour"], F4["fa_per_hour"]), "lift_minus_F": x["lift"] - F4["lift"]}
        v["R1c.4"] = {"holds": x["keep_false"] <= 0.5 * x["keep_correct"] and x["keep_correct"] >= 0.8,
                      "keep_correct": x["keep_correct"], "keep_false": x["keep_false"]}
    if g(f"GOV|{m4}"):
        x = g(f"GOV|{m4}")
        v["R1c.2"] = {"holds": x["fa_per_hour"] <= F4["fa_per_hour"] and x["lift"] >= F4["lift"] + 0.05,
                      "fa_over_F": ratio(x["fa_per_hour"], F4["fa_per_hour"]), "lift_minus_F": x["lift"] - F4["lift"]}
    if g(f"Fc|{m4}") and g(f"FV|{m4}"):
        x, y = g(f"Fc|{m4}"), g(f"FV|{m4}")
        cut = 1 - ratio(x["fa_per_hour"], F4["fa_per_hour"])
        v["R1c.3"] = {"holds": cut >= 0.25 and tok(f"Fc|{m4}", m4) <= 1.05 * tok(f"F|{m4}", m4)
                      and x["fa_per_hour"] > y["fa_per_hour"],
                      "fa_cut": cut, "token_ratio": tok(f"Fc|{m4}", m4) / tok(f"F|{m4}", m4),
                      "fc_fa": x["fa_per_hour"], "fv_fa": y["fa_per_hour"]}
    if F8 and g(f"FV|{m8}"):
        x = g(f"FV|{m8}")
        v["R1c.5"] = {"holds": x["fa_per_hour"] <= 0.5 * F8["fa_per_hour"] and abs(x["lift"] - F8["lift"]) <= 0.02,
                      "fa_ratio": ratio(x["fa_per_hour"], F8["fa_per_hour"]), "lift_minus_F": x["lift"] - F8["lift"]}
    if F8 and g("F4V8|cascade"):
        x = g("F4V8|cascade")
        v["R1c.6"] = {"holds": x["lift"] >= F8["lift"] - 0.05 and x["fa_per_hour"] <= 0.5 * F8["fa_per_hour"]
                      and x["verified_windows_share"] <= 0.5,
                      "lift_minus_8B_F": x["lift"] - F8["lift"],
                      "fa_ratio_vs_8B_F": ratio(x["fa_per_hour"], F8["fa_per_hour"]),
                      "share_8B": x["verified_windows_share"]}
    res["predictions"] = v
    print("\n=== predictions ===")
    for k, x in sorted(v.items()):
        print(f"  {k}: {'HELD' if x['holds'] else 'failed'}  " + json.dumps(
            {a: (round(b, 3) if isinstance(b, float) else b) for a, b in x.items() if a != "holds"}))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
