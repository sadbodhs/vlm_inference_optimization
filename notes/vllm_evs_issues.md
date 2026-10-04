# Draft vLLM issues: EVS (`--video-pruning-rate`)

Drafts for the maintainer to file at github.com/vllm-project/vllm/issues. Not filed
by the study. Evidence lives in this repo (paths below).

---

## Issue 1: [Bug] Qwen3-VL + EVS: every video request crashes the engine in `recompute_mrope_positions` (v0.29.0, v0.30.0)

### Your current environment
- vLLM docker images `vllm/vllm-openai:v0.29.0` and `vllm/vllm-openai:v0.30.0` (same failure)
- 1× RTX 3090 24 GB, CUDA driver of the host
- Model: `cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit` (rev `b6a6f1cc360d21feee7465f8656b3855d61e767a`)

### Describe the bug
With `--video-pruning-rate 0.75`, the first request carrying a video kills the
EngineCore. Image requests on the same server succeed. Without EVS, video
requests succeed.

```
vllm serve cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit --quantization compressed-tensors \
  --max-model-len 16384 --gpu-memory-utilization 0.80 \
  --no-enable-prefix-caching --no-enable-chunked-prefill --mm-processor-cache-gb 0 \
  --video-pruning-rate 0.75
```

Request: OpenAI chat completion with one `video_url` content part,
`data:video/jpeg;base64,<f1>,<f2>` (2 JPEG frames, 896×504) or 8 frames, plus
`"media_io_kwargs": {"video": {"fps": 1}}` (or 5). Both fail.

```
vllm/v1/worker/gpu/model_states/mm_pruning.py", line 85, in recompute
vllm/model_executor/models/qwen3_vl.py", line 2829, in recompute_mrope_positions
vllm/model_executor/models/qwen3_vl.py", line 2880, in _recompute_mrope_positions
vllm/multimodal/video_prune/evs.py", line 338, in recompute_mrope_positions
RuntimeError: The expanded size of the tensor (0) must match the existing size (428)
at non-singleton dimension 1.  Target sizes: [3, 0].  Tensor sizes: [3, 428]
```

Notes: chunked prefill is disabled, so this looks distinct from (or broader than)
#48833 / PR #48835. #44204 and #44200 were closed as not planned without a fix in
these releases. The Qwen2.5-VL path with the same flags does not crash on a
single request.

Evidence: `results/evs_v030/server-E_q3vl_4b-evs0.75.log`, `results/evs_v030.log`.

---

## Issue 2: [Bug] EVS admits video requests at their pruned size, then the vision encoder OOMs under concurrency (Qwen2.5-VL, v0.29.0, v0.30.0)

### Your current environment
- `vllm/vllm-openai:v0.29.0` and `v0.30.0`, 1× RTX 3090 24 GB
- Model: `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (rev `536a35794df8831aa814970ee8f89eff577e7718`), `--quantization awq_marlin`

### Describe the bug
With `--video-pruning-rate 0.75`, 32 concurrent requests, each one 8-frame video
(896×504 JPEG frames, fps 5), crash the engine:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 468.00 MiB ...
  in the vision encoder (qwen2_5_vl.py: Qwen2_5_VisionTransformer forward, MLP activation)
```

The scheduler had 23 requests running and KV cache only 10% used.

**Control:** the same 450 requests at the same concurrency with EVS off complete
450/450, and the server stays up. Without EVS the requests are 3.3× longer
(2,354 vs ~735 prompt tokens), so the crash is not from load in general.

### Likely cause
EVS fixes each video's placeholder count at its *pruned* size
(`compute_retained_tokens_count`), so the scheduler's encoder budget counts ~735
tokens per request. But `_process_video_input` runs the encoder on all frames
(2,304 merged tokens; 9,216 patches) before `_postprocess_video_embeds_evs` prunes.
Encoder work per scheduled request is therefore ~3× what the budget assumed. Under
load more requests' encoder inputs land in one step than memory profiling allowed
for. Expected: admission should count unpruned encoder tokens (or the encoder
should process only what it will keep).

### Minimal reproduction
```
vllm serve Qwen/Qwen2.5-VL-7B-Instruct-AWQ --quantization awq_marlin --max-model-len 16384 \
  --gpu-memory-utilization 0.80 --limit-mm-per-prompt '{"image":16}' \
  --no-enable-prefix-caching --no-enable-chunked-prefill --mm-processor-cache-gb 0 \
  --video-pruning-rate 0.75
```
Then send 450 chat requests at concurrency 32, each with one
`data:video/jpeg;base64,<8 frames>` part, `media_io_kwargs={"video":{"fps":5}}`,
`max_tokens 16`.

Evidence: `results/evs_v030/server-V_vllm_video-evs0.75.log` (crash),
`results/evs_v030/server-V_vllm_video-evs0.log` (control),
`results/e8/server-live-evs-rerun.log` (the same crash in a live 12-camera run on v0.29.0).

### Also worth noting in the EVS docs
The always-kept first frame counts toward the retained budget, so for a video of T
temporal grids any rate ≥ 1 − 1/T keeps only the first grid (e.g. 8 frames = 4
grids at rate 0.75). That surprised us, and it is easy to misread as "8 frames
pruned to the cost of 2".
