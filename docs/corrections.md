# Corrections

Seven measurement bugs found while building this. Four produced
**publishable-looking numbers that were wrong**. None were caught by inspection —
every one was caught by a physical bound, a cross-experiment contradiction, or an
internal inconsistency in the data.

This page exists because the corrections are more useful than the numbers.

## The three that would have been published

### The caching result, wrong by 8×

E0 cycled a 32-document pool, so with caching on nearly every request was a hit.
The defaults arm looked **8× faster** than the clean arm — a spectacular result,
and a measurement of a ~100% input-reuse workload that almost nobody has.

E3 said caching bought nothing on distinct inputs. E0 said 8×. Both could not be
true, **and the contradiction is what exposed it**.

Fixing it left all five rates sharing one 80-document pool, so only the first rate
was cold — visible as the two arms agreeing to within **3 ms at 0.5 QPS** while
every later rate was 13× apart.

### "Accuracy collapses at every resolution"

E3 first scored 0.018–0.030 ANLS, flat across every token budget. Read naively:
*resolution does not matter for DocVQA* — a headline result, and completely wrong.

The model was reading the documents nearly perfectly and answering in sentences:

> gold `['100001']` → pred `'The invoice number is 100001.'` → ANLS **0.00**

**92 of 100 predictions contained the gold answer.** DocVQA scores a bare span, so
the fix was one line of prompt. `tools/inspect_outputs.py` now reports how often
the gold answer appears *inside* a longer prediction, so a formatting failure can
never again be mistaken for a model failure.

### "Past the peak, more pixels are strictly worse"

At n=100 the E3 curve was non-monotonic and appeared to peak then decline. The
−0.007 ANLS that suggested it sat inside a 95% CI of [−0.013, +0.033]. At n=500 the
curve is cleanly monotonic — the decline was sampling noise.

## The three that killed a run silently

### A `grep` ended a two-hour study

`EXTRA=$(grep -E '^server_args:' "$ARM" | ...)` — grep exits 1 when the field is
absent, `pipefail` propagates it, `set -e` kills the script. The clean arm defines
`server_args` so it survived; the defaults arm does not, so the study aborted the
moment it reached it, with **no error in the log**.

### A `pgrep` that matched itself

The liveness check was `pgrep -f run_full_study` — run over SSH, in a command
string that *contains* `run_full_study`. It matched itself and reported the study
alive for **2h21m** after it had died.

### A harness that reported its own pacing as capacity

E4's tick loop awaited every analysis before starting the next, pinning concurrency
at the stream count. It produced a flat **8.18 analyses/s across an 8× range of
demand** — which reads as clean saturation and is really 8 requests at ~980 ms
each. `bench/loadgen.py` warns about exactly this; E4 was written with the flaw the
harness was built to avoid.

## The one that was a modelling error, not a code bug

The roofline first used the **total** 7.16 GB weight footprint. Decode re-reads only
the LLM weights — the bf16 vision tower runs once in prefill — so the real basis is
5.81 GB. Against the wrong basis, the measured 149.4 tok/s read as **114% of
roofline**: impossible. The gate fired correctly, on a wrong model rather than
wrong code.

## The hypothesis that was tested and failed

The SGLang accuracy gap grew with vision-token count, which is the signature of
numerical divergence accumulating in the vision tower — and SGLang defaults to a
different vision attention kernel than vLLM. A clean mechanism, fitting the data.

Pinning both stacks to the same backend changed the gap by 0.002. The mechanism
was wrong.

It is on this page because a plausible story that fits the data is not evidence,
and the only thing separating this from the four entries above is that it was
tested before being published.

## The pattern

Every one of these **reported plausible success while doing something other than
what it claimed**. That is the failure mode a benchmark harness has to be designed
against, because it is invisible to code review and indistinguishable from a
result.

The three mechanisms that actually caught them:

1. **Physical bounds** — a decode rate above the memory-bandwidth roofline cannot exist
2. **Cross-experiment contradiction** — E3 and E0 disagreeing about caching by 8×
3. **Internal consistency** — a cold rate matching the no-cache arm to within 3 ms

None of these require a profiler, and none of them are in the code being tested.
