"""Per-request record. One row per request in latency.jsonl."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class RequestResult:
    arm_id: str
    sample_id: str

    # Timeline, all from a single perf_counter origin (seconds).
    scheduled_s: float          # when the load generator intended to send
    sent_s: float               # when it actually sent
    first_token_s: float | None = None
    last_token_s: float | None = None

    itl_ms: list[float] = field(default_factory=list)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Vision tokens are not reported by any serving stack; derived in calibration.
    vision_tokens: int | None = None

    image_px: int | None = None
    text: str = ""
    ok: bool = True
    error: str | None = None

    # ---- derived -------------------------------------------------------

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_s is None:
            return None
        return (self.first_token_s - self.sent_s) * 1e3

    @property
    def e2e_ms(self) -> float | None:
        if self.last_token_s is None:
            return None
        return (self.last_token_s - self.sent_s) * 1e3

    @property
    def tpot_ms(self) -> float | None:
        """Mean inter-token latency, excluding the first token."""
        if not self.itl_ms:
            return None
        return sum(self.itl_ms) / len(self.itl_ms)

    @property
    def loadgen_lag_ms(self) -> float:
        """How late the client was in issuing this request.

        This is the R2 tripwire. If this grows over a run, the load generator --
        not the server -- is the bottleneck, and every throughput number from the
        run is a measurement of the harness.
        """
        return (self.sent_s - self.scheduled_s) * 1e3

    @property
    def decode_tok_s(self) -> float | None:
        if self.first_token_s is None or self.last_token_s is None:
            return None
        span = self.last_token_s - self.first_token_s
        if span <= 0 or not self.itl_ms:
            return None
        return len(self.itl_ms) / span

    def to_row(self) -> dict:
        d = dataclasses.asdict(self)
        d.pop("text", None)  # kept separately; keeps latency.jsonl small
        d.update(
            ttft_ms=self.ttft_ms,
            e2e_ms=self.e2e_ms,
            tpot_ms=self.tpot_ms,
            loadgen_lag_ms=self.loadgen_lag_ms,
            decode_tok_s=self.decode_tok_s,
            n_itl=len(self.itl_ms),
        )
        d["itl_ms"] = [round(x, 4) for x in self.itl_ms]
        return d
