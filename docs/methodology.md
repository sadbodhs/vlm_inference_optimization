# Methodology

The [bar for a number here](https://sadbodhs.github.io/) is enforced in code rather
than in prose.

## R1 · One variable at a time

Every stack is driven through one OpenAI-compatible streaming path. An arm is a
YAML file; adding one never adds a branch.

`docker/run_server.sh` dispatches on the arm's `stack:` field, so swapping vLLM for
SGLang is a config edit, not a different script. Everything else — harness image,
prompts, images, scorer — is byte-identical across arms.

!!! warning "Where cross-stack comparisons actually break"
    Not in the measurement. In the *configuration*. Three separate times a
    comparison was set up with what looked like matched settings and wasn't:
    each vendor's "defaults off" meant a different set of defaults; chunked
    prefill was on in one stack and off in the other; and
    `--mem-fraction-static 0.90` turned out not to be the analogue of
    `--gpu-memory-utilization 0.90`. Each produced plausible numbers.

## R2 · Saturate before you measure

Every run self-reports `saturated`, `not-saturated`, or **`client-bound`**.

The third is the one nobody checks. If the load generator falls behind schedule,
the throughput number describes the harness, not the server. It is detected by
tracking per-request scheduling lag drift, and such runs are discarded.

Saturation itself is judged on *queue growth*, not on achieved/offered: Poisson
arrival variance at n=32 is ±18%, enough to manufacture a saturation signal.

Load is generated **open loop** — arrivals fire on a schedule regardless of
completion. A closed loop cannot produce a queue, so it cannot show queueing delay,
and it silently rate-limits itself to whatever the server can do.

## R3 · Publish the cost

Accuracy is scored from the same generated outputs as the latency run, so a
frontier point is never assembled from two experiments.

## R4 · Say what was not measured

Every run writes a `not_measured` list into `meta.json`. See
[Not measured](not-measured.md).

## Provenance

Every result carries: harness image digest, server image digest, model revision,
git SHA (plus a dirty flag), and GPU state **sampled during the run** — not
before and after, because the end probe usually catches the recovery rather than
the load.

## Guards that make a wrong number loud

| guard | catches |
|---|---|
| roofline gate | a decode rate above the memory-bandwidth ceiling — physically impossible, so it can only be a harness bug |
| all-failed check | a run where every request errored returning quietly as empty columns |
| concurrency guard | two experiments hitting one server and contaminating each other |
| containment check | the model answering correctly in the wrong *shape* (see [corrections](corrections.md)) |
| reuse accounting | a cache silently serving a workload that was supposed to be cold |
