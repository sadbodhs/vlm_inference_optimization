"""Arms are data. An arm is a YAML file; adding one must never add a branch."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_yaml(text: str) -> dict:
    try:
        import yaml
        return yaml.safe_load(text)
    except ImportError:
        from ruamel.yaml import YAML
        import io
        return YAML(typ="safe").load(io.StringIO(text))


@dataclass
class Arm:
    id: str
    model: str
    base_url: str = "http://127.0.0.1:8000"
    stack: str = "vllm"
    quantization: str | None = None
    notes: str = ""

    # request-shaping knobs
    max_tokens: int = 128
    temperature: float = 0.0
    extra_body: dict[str, Any] = field(default_factory=dict)

    # the Phase-5 independent variable; None means "stack default", which is a
    # trap on dynamic-resolution models -- see PLAN.md 3.
    max_pixels: int | None = None

    # Total weight footprint in VRAM. Reported by the server at load; used for the
    # memory budget, NOT for the decode roofline.
    weight_bytes: float | None = None

    # Weights re-read on EVERY decode step. This excludes the vision tower, which
    # runs once during prefill and is never touched again while tokens stream. Using
    # the total footprint here understates the decode ceiling by the size of the
    # ViT -- on a 7B VLM that is over a GiB, and it turns a passing roofline check
    # into a failing one. Falls back to weight_bytes when unset.
    decode_weight_bytes: float | None = None

    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # PyYAML follows YAML 1.1, where `6.0e9` is a STRING -- an exponent needs a
        # sign (`6.0e+9`) to parse as a float. ruamel follows YAML 1.2 and parses it
        # as a number. Coerce here so an arm behaves identically whichever loader is
        # present, rather than failing only inside the container.
        for f in ("weight_bytes", "decode_weight_bytes"):
            v = getattr(self, f)
            if isinstance(v, str):
                setattr(self, f, float(v))
        for f in ("max_pixels", "max_tokens"):
            v = getattr(self, f)
            if isinstance(v, str):
                setattr(self, f, int(float(v)))
        if isinstance(self.temperature, str):
            self.temperature = float(self.temperature)

    @classmethod
    def load(cls, path: str | Path) -> "Arm":
        path = Path(path)
        raw = _load_yaml(path.read_text()) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "raw"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs.setdefault("id", path.stem)
        return cls(raw=raw, **kwargs)

    @property
    def roofline_bytes(self) -> float | None:
        return self.decode_weight_bytes or self.weight_bytes

    def request_extra(self) -> dict:
        extra = dict(self.extra_body)
        if self.max_pixels is not None:
            # vLLM/SGLang route per-request multimodal limits through here.
            extra.setdefault("mm_processor_kwargs", {})["max_pixels"] = self.max_pixels
        return extra
