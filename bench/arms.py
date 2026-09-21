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

    # declared, not measured: used only for the roofline, and echoed into meta.json
    weight_bytes: float | None = None

    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Arm":
        path = Path(path)
        raw = _load_yaml(path.read_text()) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "raw"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        kwargs.setdefault("id", path.stem)
        return cls(raw=raw, **kwargs)

    def request_extra(self) -> dict:
        extra = dict(self.extra_body)
        if self.max_pixels is not None:
            # vLLM/SGLang route per-request multimodal limits through here.
            extra.setdefault("mm_processor_kwargs", {})["max_pixels"] = self.max_pixels
        return extra
