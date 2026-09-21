#!/usr/bin/env python3
"""E3 -- the token-budget Pareto frontier: what accuracy costs what latency.

This is the thesis experiment. E2 established that TTFT is roughly linear in vision
tokens; E3 asks the only question that matters after that -- how far down can you
turn the token budget before the answers stop being right, and does that limit
depend on the task?

Sweeps `max_pixels` and, for each budget, measures TTFT *and* task accuracy from
the same generated outputs. Run it once per dataset. The expected shape:

  DocVQA   accuracy falls off a cliff once text stops being resolvable
  ChartQA  intermediate
  VQAv2    nearly flat -- most of the token budget is being wasted

If that holds, "how many vision tokens do I need" has no single answer, and any
paper reporting one number is reporting its dataset. That is the result.

R3: accuracy, TTFT and the token count are reported together or not at all.

    python3 experiments/e3_token_budget.py --arm arms/B_vllm_awq.yaml \
        --manifest data/docvqa/manifest.jsonl --scorer anls
"""
from __future__ import annotations

import asyncio

from _common import base_parser, resolve_arm, table, write_sweep


async def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--manifest", default=None,
                   help="JSONL manifest; omit to use synthetic images (no accuracy)")
    p.add_argument("--scorer", default="anls", choices=["anls", "relaxed", "em"])
    p.add_argument("--budgets", default="200704,451584,802816,1605632,3211264",
                   help="max_pixels values, comma separated")
    p.add_argument("--limit", type=int, default=100, help="samples per budget")
    p.add_argument("--max-tokens", type=int, default=64)
    # DocVQA/ChartQA metrics score an exact span. Without this instruction the model
    # answers correctly in a full sentence and ANLS scores it zero: measured on this
    # rig, 92/100 predictions contained the gold answer while ANLS read 0.03. That is
    # a prompt artefact masquerading as a catastrophic accuracy result, and it made
    # every token budget look equally useless.
    p.add_argument("--prompt-suffix",
                   default="\nAnswer the question using a single word or phrase.")
    args = p.parse_args()

    from bench.data import load_manifest, synthetic
    from bench.harness import run_arm

    arm = resolve_arm(args)
    arm.max_tokens = args.max_tokens
    tag = args.tag or f"e3-budget-{arm.id}"
    budgets = [int(x) for x in args.budgets.split(",")]

    if args.manifest:
        samples = load_manifest(args.manifest, limit=args.limit)
        scorer = args.scorer
        dataset = args.manifest
    else:
        samples = synthetic(min(args.limit, 16), width=1344, height=1344)
        scorer = None
        dataset = "synthetic"
        print("no --manifest: running the latency axis only. Accuracy will be null,\n"
              "which makes this a shape check, not a Pareto frontier.")

    rows = []
    for budget in budgets:
        arm.max_pixels = budget
        s = await run_arm(
            arm, samples, mode="sequential", scorer=scorer,
            prompt_suffix=args.prompt_suffix,
            results_root=args.out, run_id=f"{tag}-px{budget}",
            unmeasured=(["accuracy"] if not scorer else [])
            + ["throughput under load", "multi-image prompts"],
        )
        rows.append({
            "max_pixels": budget,
            "max_edge_sq": int(budget**0.5),
            "prompt_tokens": s["prompt_tokens"]["mean"],
            "ttft_p50_ms": s["ttft_ms"]["p50"],
            "ttft_p95_ms": s["ttft_ms"]["p95"],
            "accuracy": s["accuracy"]["score"],
            "n_scored": s["accuracy"].get("n_scored"),
        })
        acc = rows[-1]["accuracy"]
        print(f"  max_pixels={budget:<9} TTFT p50={s['ttft_ms']['p50']:>7.1f} ms   "
              f"acc={'n/a' if acc is None else f'{acc:.3f}'}")

    print(f"\nE3 · token budget vs accuracy   dataset={dataset}  scorer={scorer}")
    print(f"   prompt suffix: {args.prompt_suffix!r}")
    table(rows, [("max_pixels", "max_pixels"), ("prompt_tokens", "prompt tok"),
                 ("ttft_p50_ms", "TTFT p50"), ("accuracy", "accuracy"),
                 ("n_scored", "n scored")])

    scored = [r for r in rows if r["accuracy"] is not None]
    if len(scored) >= 2:
        best = max(scored, key=lambda r: r["accuracy"])
        print(f"\n  peak accuracy {best['accuracy']:.3f} at max_pixels="
              f"{best['max_pixels']} ({best['ttft_p50_ms']:.0f} ms)")

        # A single "knee" hides the shape. What a practitioner actually asks is:
        # how much accuracy am I willing to give up, and what does that buy?
        print("\n  cheapest budget within N points of peak:")
        for tol in (0.01, 0.02, 0.03, 0.05):
            cands = [r for r in scored if r["accuracy"] >= best["accuracy"] - tol]
            if not cands:
                continue
            k = min(cands, key=lambda r: r["ttft_p50_ms"])
            saved = 100.0 * (1 - k["ttft_p50_ms"] / best["ttft_p50_ms"])
            print(f"    -{tol*100:.0f} pts: max_pixels={k['max_pixels']:<8} "
                  f"acc={k['accuracy']:.3f}  TTFT {k['ttft_p50_ms']:.0f} ms "
                  f"({saved:+.0f}% vs peak)")

        # Past the peak, more tokens cost time and buy nothing -- or hurt. That is
        # the finding that justifies capping max_pixels at all.
        over = [r for r in scored if r["max_pixels"] > best["max_pixels"]]
        for r in over:
            dt = 100.0 * (r["ttft_p50_ms"] / best["ttft_p50_ms"] - 1)
            da = r["accuracy"] - best["accuracy"]
            print(f"\n  beyond peak: max_pixels={r['max_pixels']} costs {dt:+.0f}% TTFT "
                  f"for {da:+.3f} accuracy")

    write_sweep(args.out, tag, rows,
                {"experiment": "e3-token-budget", "arm": arm.id, "dataset": dataset,
                 "scorer": scorer, "budgets": budgets, "limit": args.limit})


if __name__ == "__main__":
    asyncio.run(main())
