"""Eval samples and image encoding.

Datasets are frozen local subsets described by a JSONL manifest, so a run is
reproducible without network access. `synthetic` generates images of an exact pixel
count, which is what the token-budget experiments actually need -- there the image
*content* is irrelevant and only its geometry drives cost.
"""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sample:
    id: str
    question: str
    answers: list[str] = field(default_factory=list)
    image_path: str | None = None
    image_b64: str | None = None
    image_mime: str = "png"
    image_px: int | None = None
    # Video: several frames in one prompt. A clip costs frames x tokens-per-frame,
    # so these two are in direct competition inside a fixed context budget.
    frames_b64: list[str] = field(default_factory=list)

    def data_url(self) -> str:
        if self.image_b64:
            return f"data:image/{self.image_mime};base64,{self.image_b64}"
        raw = Path(self.image_path).read_bytes()
        suffix = Path(self.image_path).suffix.lstrip(".").lower() or "png"
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        return f"data:image/{mime};base64,{base64.b64encode(raw).decode()}"

    def with_frames(self, frames: list[bytes], max_pixels: int | None,
                    mime: str = "jpeg") -> "Sample":
        """Copy carrying `frames`, each re-encoded within the per-frame budget.

        Done up front, outside the timed path: decoding and resizing N frames
        inside the request loop would bill video preprocessing to the server's TTFT.
        """
        import base64 as _b64

        from .imaging import resize_to_budget

        out, px = [], 0
        for raw in frames:
            data, _, h, w = resize_to_budget(raw, max_pixels)
            out.append(_b64.b64encode(data).decode())
            px += h * w
        return Sample(id=self.id, question=self.question, answers=list(self.answers),
                      frames_b64=out, image_mime=mime, image_px=px)

    def resized(self, max_pixels: int | None) -> "Sample":
        """Copy of this sample with the image re-encoded within a pixel budget.

        Done once, up front, outside the timed path -- resizing inside the request
        loop would bill image processing to the server's TTFT.
        """
        from .imaging import resize_to_budget

        raw = (base64.b64decode(self.image_b64) if self.image_b64
               else Path(self.image_path).read_bytes())
        data, mime, h, w = resize_to_budget(raw, max_pixels)
        return Sample(
            id=self.id, question=self.question, answers=list(self.answers),
            image_path=None, image_b64=base64.b64encode(data).decode(),
            image_mime=mime, image_px=h * w,
        )

    def to_messages(self, prompt_suffix: str = "") -> list[dict]:
        content: list[dict] = []
        if self.frames_b64:
            # Frames first, in temporal order. Anything asking about direction,
            # speed or ordering depends on that order being preserved.
            for f in self.frames_b64:
                content.append({"type": "image_url", "image_url": {
                    "url": f"data:image/{self.image_mime};base64,{f}"}})
        elif self.image_path or self.image_b64:
            content.append({"type": "image_url",
                            "image_url": {"url": self.data_url()}})
        content.append({"type": "text", "text": self.question + prompt_suffix})
        return [{"role": "user", "content": content}]

    @property
    def n_images(self) -> int:
        return len(self.frames_b64) or (1 if (self.image_b64 or self.image_path) else 0)


def load_manifest(path: str | Path, limit: int | None = None) -> list[Sample]:
    """JSONL: {"id":..., "question":..., "answers":[...], "image": "rel/path.png"}"""
    path = Path(path)
    root = path.parent
    out: list[Sample] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        img = rec.get("image")
        out.append(
            Sample(
                id=str(rec["id"]),
                question=rec["question"],
                answers=rec.get("answers", []),
                image_path=str(root / img) if img else None,
            )
        )
        if limit and len(out) >= limit:
            break
    return out


def synthetic(n: int, *, width: int, height: int, seed: int = 0,
              question: str = "What is written in this document?") -> list[Sample]:
    """n images of exactly width x height. Content is noise-free structured text-like
    bars: the point is geometry, not legibility.

    `seed` genuinely changes the pixels, and must. Repeated runs that re-send
    identical images are independent samples only while every cache is off; with a
    multimodal cache enabled they become cache hits, and averaging them reports a
    blend of one cold measurement and N-1 warm ones as if it were a spread.
    """
    from PIL import Image, ImageDraw

    out: list[Sample] = []
    for i in range(n):
        img = Image.new("RGB", (width, height), (255, 255, 255))
        d = ImageDraw.Draw(img)
        step = max(height // 24, 4)
        jitter = (i * 37 + seed * 9173) % max(width // 3, 1)
        for k, y in enumerate(range(step, height - step, step * 2)):
            right = width - width // 20 - ((jitter + k * 13 * (seed + 1)) % max(width // 3, 1))
            d.rectangle([width // 20, y, max(right, width // 10), y + step // 2],
                        fill=(40, 40, 40))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(
            Sample(
                id=f"syn-{width}x{height}-s{seed}-{i}",
                question=question,
                image_b64=base64.b64encode(buf.getvalue()).decode(),
                image_px=width * height,
            )
        )
    return out


def text_only(n: int, question: str = "What is written in this document?") -> list[Sample]:
    """Image-free controls. Needed to derive vision-token count by subtraction:
    no serving stack reports it directly."""
    return [Sample(id=f"txt-{i}", question=question) for i in range(n)]
