#!/usr/bin/env python3
"""Freeze a small evaluation subset to disk as images + a manifest.

Benchmarks that stream a dataset at run time are not reproducible: the sample
order, the preprocessing and the upstream revision can all move underneath you.
This writes a fixed subset once, so every arm is scored on byte-identical inputs
(R1) and a run needs no network.

    python3 tools/build_manifest.py --dataset lmms-lab/DocVQA --config DocVQA \\
        --split validation --n 200 --out data/docvqa

Field names differ across dataset cards, so columns are detected rather than
assumed, and the detection is printed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

Q_KEYS = ("question", "query", "prompt")
A_KEYS = ("answers", "answer", "label", "answer_text")
ID_KEYS = ("questionId", "question_id", "qid", "id")
IMG_KEYS = ("image", "img", "image_1")


def _pick(cols: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _answer(v) -> str:
    # VQAv2 stores each annotator's answer as a dict:
    #   {'answer': 'double decker', 'answer_confidence': 'maybe', 'answer_id': 4}
    # str() on that gives a Python repr no prediction can ever equal -- which is
    # how a 7B model scored exactly 0.000 at every budget.
    if isinstance(v, dict):
        for k in ("answer", "text", "label"):
            if k in v:
                return str(v[k])
        raise ValueError(f"answer dict with no recognised text field: {v!r}")
    return str(v)


def _answers(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        return [val]
    if isinstance(val, (list, tuple)):
        return [_answer(v) for v in val]
    return [_answer(val)]


def _check_answers(rows) -> None:
    """Refuse to write a manifest whose golds are serialised containers.

    A gold like "{'answer': ...}" or "['a', 'b']" is never a real answer; it
    means a structured field was stringified. Scoring against it yields a clean,
    plausible-looking 0.0 rather than an error, so it has to be caught here."""
    bad = [(r["id"], a) for r in rows for a in r["answers"]
           if a[:1] in "{[" and a[-1:] in "}]"]
    if bad:
        sid, a = bad[0]
        raise SystemExit(f"{len(bad)} gold answers look like serialised containers, "
                         f"e.g. sample {sid}: {a[:80]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, args.config, split=args.split)
    cols = list(ds.column_names)
    q_k, a_k, i_k = _pick(cols, Q_KEYS), _pick(cols, A_KEYS), _pick(cols, IMG_KEYS)
    id_k = _pick(cols, ID_KEYS)
    print(f"columns: {cols}")
    print(f"detected -> question={q_k!r} answers={a_k!r} image={i_k!r} id={id_k!r}")
    if not (q_k and i_k):
        raise SystemExit("could not find a question and an image column; pass a "
                         "different dataset or extend the key lists in this file")

    # Deterministic subset: shuffle with a fixed seed, then take the first n, so
    # the subset is reproducible but not just the head of the file.
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    n_no_answer = 0
    for i, rec in enumerate(ds):
        img = rec[i_k]
        if img is None:
            continue
        sid = str(rec.get(id_k, i)) if id_k else str(i)
        rel = f"images/{i:05d}.png"
        img.convert("RGB").save(out / rel)
        answers = _answers(rec.get(a_k)) if a_k else []
        n_no_answer += not answers
        rows.append({"id": sid, "question": rec[q_k], "answers": answers, "image": rel})

    _check_answers(rows)
    (out / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    meta = {"dataset": args.dataset, "config": args.config, "split": args.split,
            "n": len(rows), "seed": args.seed, "columns": cols,
            "detected": {"question": q_k, "answers": a_k, "image": i_k, "id": id_k},
            "n_without_answers": n_no_answer}
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"wrote {len(rows)} samples to {out}/manifest.jsonl")
    if n_no_answer:
        # A test split with hidden labels scores 0.0 everywhere and looks like a
        # catastrophic accuracy result rather than a missing-label problem.
        print(f"  WARNING: {n_no_answer}/{len(rows)} samples have no answers. "
              f"Use a split with public labels or E3's accuracy axis is meaningless.")


if __name__ == "__main__":
    main()
