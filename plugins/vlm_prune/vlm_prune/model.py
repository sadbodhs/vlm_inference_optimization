"""Pre-encoder EVS for Qwen2.5-VL (E9).

vLLM's EVS (--video-pruning-rate q) runs the vision encoder on every frame pair,
then keeps the (1 - q) share of output tokens that changed most since the previous
pair (the first pair always whole). E8 measured what that buys: shorter prompts,
the same encoder work, no extra cameras -- and an encoder OOM under load, because
requests are admitted at their pruned size.

This subclass keeps vLLM's contract -- the same number of tokens, so the prompt
placeholders, M-RoPE positions and scheduler accounting are untouched -- and
changes two things:

  1. which tokens: ranked by PIXEL change of the merged 28 px unit against the
     same unit of the previous frame pair (mean |difference| of its 4 normalised
     patches, both frames of the pair), not by encoder-output similarity;
  2. what the encoder sees: ONLY the kept units. Qwen2.5-VL's window attention and
     per-frame full attention are rebuilt for the subset, so every kept patch
     keeps its true rotary position, window and frame.

PEVS_CHECK=1 also runs the full encoder per video and logs, every 50 videos, the
cosine similarity of the kept tokens to their full-encoder values and the time of
both encoder passes. It doubles encoder work: development only.
"""
from __future__ import annotations

import os
import time

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VLDummyInputsBuilder,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLMultiModalProcessor,
    Qwen2_5_VLProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.video_prune.evs import (
    compute_mrope_for_media,
    compute_retained_tokens_count,
)
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

logger = init_logger("vllm.vlm_prune")   # only "vllm.*" loggers are printed
CHECK = os.environ.get("PEVS_CHECK", "0") == "1"
# unit: rank single 28 px units; window: rank whole attention windows (4 x 4 units,
# 112 px) by their most-changed unit, so a kept unit keeps its window-mates
MODE = os.environ.get("PEVS_MODE", "unit")


def keep_units_by_pixels(pixels: torch.Tensor, t: int, units_per_t: int, q: float,
                         grid_hw: tuple[int, int] | None = None, win: int = 4) -> torch.Tensor:
    """Bool mask over merged units in (t, h/2, w/2) raster order: the first frame
    pair whole, then the most-changed units of later pairs, EVS's exact count."""
    u = pixels.reshape(t, units_per_t, -1).float()
    keep_n = compute_retained_tokens_count(units_per_t, t, q)
    mask = torch.zeros(t * units_per_t, dtype=torch.bool, device=pixels.device)
    mask[:units_per_t] = True
    extra = keep_n - units_per_t
    if extra > 0 and t > 1:
        score = (u[1:] - u[:-1]).abs().mean(-1)                    # (t-1, units)
        if MODE == "window" and grid_hw is not None:
            gh, gw = grid_hw
            s = score.reshape(t - 1, gh, gw)
            ph, pw = (-gh) % win, (-gw) % win
            wmax = F.max_pool2d(F.pad(s, (0, pw, 0, ph)).unsqueeze(1), win).squeeze(1)
            wmax = wmax.repeat_interleave(win, 1).repeat_interleave(win, 2)[:, :gh, :gw]
            score = wmax + 1e-3 * s                 # whole windows first, units break ties
        score = score.reshape(-1)
        top = torch.argsort(score, descending=True, stable=True)[:extra]
        mask[units_per_t + top] = True
    return mask


def subset_metadata(visual, grid, keep: torch.Tensor):
    """Encoder metadata for only the kept units of one video item.

    Returns (metadata, patch_rows): patch_rows index the item's pixel rows of the
    kept units in raster order -- the order visual.forward expects its input in.
    """
    t, h, w = (int(v) for v in grid)
    m = visual.spatial_merge_unit                       # 4 patches per unit
    units_per_t = (h // visual.spatial_merge_size) * (w // visual.spatial_merge_size)
    dev = keep.device
    cos, sin, widx, cuw, _ = visual.get_rope_by_thw(t, h, w)
    widx = widx.to(dev)
    keep_w = keep[widx]                                 # kept, in window order
    pos = torch.nonzero(keep_w).squeeze(1)
    pidx = (pos[:, None] * m + torch.arange(m, device=dev)).reshape(-1)
    cos_k, sin_k = cos.to(dev)[pidx], sin.to(dev)[pidx]

    # window boundaries: units per window from the full cumulative patch counts
    cuw_full = torch.unique_consecutive(torch.cat([torch.zeros(1, dtype=cuw.dtype), cuw.cpu()]))
    win_units = torch.diff(cuw_full) // m
    win_id = torch.repeat_interleave(torch.arange(len(win_units)), win_units).to(dev)
    kept_win = torch.bincount(win_id[keep_w], minlength=len(win_units)) * m
    cu_win = torch.unique_consecutive(F.pad(torch.cumsum(kept_win, 0), (1, 0))).to(torch.int32)
    # full attention is per frame pair; window order is frame-major
    frame_id = widx // units_per_t
    kept_frame = torch.bincount(frame_id[keep_w], minlength=t) * m
    cu_full = torch.unique_consecutive(F.pad(torch.cumsum(kept_frame, 0), (1, 0))).to(torch.int32)

    raster = torch.nonzero(keep).squeeze(1)             # kept units, raster order
    rank = torch.full((keep.numel(),), -1, dtype=torch.long, device=dev)
    rank[raster] = torch.arange(raster.numel(), device=dev)
    window_index = rank[widx[keep_w]]
    reverse = torch.empty_like(window_index)
    reverse[window_index] = torch.arange(window_index.numel(), device=dev)

    be, md = visual.attn_backend, {}
    cu_full_np, cu_win_np = cu_full.cpu().numpy(), cu_win.cpu().numpy()
    md["rotary_pos_emb_cos"], md["rotary_pos_emb_sin"] = cos_k, sin_k
    md["window_index"], md["reverse_indices"] = window_index, reverse
    md["sequence_lengths_full"] = MMEncoderAttention.maybe_compute_seq_lens(be, cu_full_np, dev)
    md["sequence_lengths_window"] = MMEncoderAttention.maybe_compute_seq_lens(be, cu_win_np, dev)
    md["max_seqlen_full"] = torch.tensor(MMEncoderAttention.compute_max_seqlen(be, cu_full_np), dtype=torch.int32)
    md["max_seqlen_window"] = torch.tensor(MMEncoderAttention.compute_max_seqlen(be, cu_win_np), dtype=torch.int32)
    md["cu_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
        be, cu_full_np, visual.hidden_size, visual.tp_size, dev,
        fp8_padded_hidden_size=visual.fp8_padded_hidden_size)
    md["cu_window_seqlens"] = MMEncoderAttention.maybe_recompute_cu_seqlens(
        be, cu_win_np, visual.hidden_size, visual.tp_size, dev,
        fp8_padded_hidden_size=visual.fp8_padded_hidden_size)
    rows = (raster[:, None] * m + torch.arange(m, device=dev)).reshape(-1)
    return md, rows


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5_VLMultiModalProcessor,
    info=Qwen2_5_VLProcessingInfo,
    dummy_inputs=Qwen2_5_VLDummyInputsBuilder,
)
class PrunedQwen2_5_VL(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._pevs_n = 0
        self._pevs_stats: list[tuple[float, float, float, float]] = []
        logger.info("vlm_prune: pre-encoder video pruning active (rate %s, mode %s, check %s)",
                    self.video_pruning_rate, MODE, CHECK)

    def _pre_encoder_prune(self, video_input) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        pixels_all = self.input_norm(video_input["pixel_values_videos"], self.visual.dtype)
        second = video_input.get("second_per_grid_ts")
        if second is None:
            raise ValueError("second_per_grid_ts is required for video pruning")
        second = second.long()
        merge = self.visual.spatial_merge_size
        tps = self.config.vision_config.tokens_per_second
        out, start = [], 0
        for grid, sec in zip(grid_thw.tolist(), second):
            t, h, w = grid
            n = t * h * w
            pixels = pixels_all[start:start + n]
            start += n
            units_per_t = (h // merge) * (w // merge)
            with gpu_sync_allowed():
                keep = keep_units_by_pixels(pixels, t, units_per_t, self.video_pruning_rate,
                                            (h // merge, w // merge),
                                            self.visual.window_size // self.visual.patch_size // merge)
                md, rows = subset_metadata(self.visual, grid, keep)
                if CHECK:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                emb = self.visual(pixels[rows], grid_thw=None, encoder_metadata=md)
                if CHECK:
                    torch.cuda.synchronize(); t1 = time.perf_counter()
                    full = self.visual(pixels, grid_thw=[grid])
                    torch.cuda.synchronize(); t2 = time.perf_counter()
                    cos = F.cosine_similarity(emb.float(), full[keep].float(), dim=-1)
                    # identity test: the subset path with every unit kept must equal the full path
                    md_all, rows_all = subset_metadata(self.visual, grid, torch.ones_like(keep))
                    same = self.visual(pixels[rows_all], grid_thw=None, encoder_metadata=md_all)
                    cos_all = F.cosine_similarity(same.float(), full.float(), dim=-1)
                    self._pevs_stats.append((cos.mean().item(), cos.min().item(), t1 - t0, t2 - t1,
                                             cos[:units_per_t].mean().item(),
                                             cos[units_per_t:].mean().item() if cos.numel() > units_per_t else 1.0,
                                             cos_all.min().item()))
                    self._pevs_n += 1
                    if self._pevs_n % 50 == 0:
                        s = torch.tensor(self._pevs_stats)
                        logger.info("PEVS_CHECK n=%d kept=%d/%d cos mean %.4f p05 %.4f min %.4f "
                                    "| first pair %.4f, later pairs %.4f | keep-all identity min %.5f "
                                    "| encoder subset %.1f ms full %.1f ms",
                                    self._pevs_n, int(keep.sum()), keep.numel(),
                                    s[:, 0].mean(), s[:, 0].quantile(0.05), s[:, 1].min(),
                                    s[:, 4].mean(), s[:, 5].mean(), s[:, 6].min(),
                                    1e3 * s[:, 2].median(), 1e3 * s[:, 3].median())
                        self._pevs_stats.clear()
                positions = compute_mrope_for_media(
                    grid, merge, tokens_per_second=tps,
                    video_second_per_grid=sec.item()).to(emb.device, non_blocking=True)
                positions = positions[keep]
            out.append(torch.cat([emb, positions], dim=1))
        return tuple(out)

    def embed_multimodal(self, **kwargs: object):
        if not self.is_multimodal_pruning_enabled:
            return super().embed_multimodal(**kwargs)
        mm = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm:
            return []
        out: tuple[torch.Tensor, ...] = ()
        for modality in mm:                      # dict order = prompt order
            inp = mm[modality]
            if modality == "image":
                e = self._process_image_input(inp)
                out += tuple(self._postprocess_image_embeds_evs(e, inp))
            if modality == "video":
                if inp["type"] == "video_embeds":
                    e = self._process_video_input(inp)
                    out += tuple(self._postprocess_video_embeds_evs(e, inp))
                else:
                    out += self._pre_encoder_prune(inp)
        return out
