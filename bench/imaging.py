"""Client-side image resizing, so the token budget is what we SEND.

E3's independent variable was originally set by asking the server to cap pixels
(vLLM's mm_processor_kwargs.max_pixels). That works on vLLM and is silently
ignored by SGLang: both budgets produced an identical 4,568 prompt tokens, because
SGLang processed the image at native resolution either way. An experiment whose
independent variable only exists on one stack cannot compare stacks.

Resizing before the request makes the budget stack-independent, removes the
server's preprocessing policy as a hidden variable, and measures the decision a
deployment actually makes: how many pixels to send.

The geometry mirrors Qwen2-VL's own smart_resize -- dimensions rounded to a
multiple of 28 (14px patches, 2x2 merged), aspect ratio preserved -- so the
resulting token count matches what the server would have produced itself.
"""
from __future__ import annotations

import io
import math

FACTOR = 28            # 14px patch x 2x2 spatial merge
MIN_PIXELS = 56 * 56
MAX_RATIO = 200


def smart_resize(height: int, width: int, factor: int = FACTOR,
                 min_pixels: int = MIN_PIXELS,
                 max_pixels: int | None = None) -> tuple[int, int]:
    """Target (height, width): multiples of `factor`, aspect preserved, within budget."""
    if max(height, width) / max(min(height, width), 1) > MAX_RATIO:
        raise ValueError(f"aspect ratio {height}x{width} exceeds {MAX_RATIO}")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if max_pixels and h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def vision_tokens_for(height: int, width: int) -> int:
    """Token count the model will see for an already-resized image."""
    return (height // FACTOR) * (width // FACTOR)


def resize_to_budget(raw: bytes, max_pixels: int | None,
                     quality: int = 90) -> tuple[bytes, str, int, int]:
    """Re-encode `raw` within the pixel budget. Returns (bytes, mime, h, w).

    JPEG out: a resized document is photographic in character, and PNG of the same
    image is several times larger for no benefit to the model -- payload size is
    part of what a live pipeline pays.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(raw))
    img = img.convert("RGB")
    h, w = img.height, img.width
    th, tw = smart_resize(h, w, max_pixels=max_pixels)
    if (th, tw) != (h, w):
        img = img.resize((tw, th), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), "jpeg", th, tw
