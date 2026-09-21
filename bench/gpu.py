"""GPU state probe. Thermal throttling is on the list of things nobody benchmarks,
so temperature and clocks are recorded with every run rather than assumed stable.

Returns None-filled records off-GPU (e.g. the Mac Studio) instead of raising, so the
harness stays runnable for dry runs.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time

_FIELDS = [
    "name", "memory.total", "memory.used", "temperature.gpu",
    "clocks.sm", "clocks.max.sm", "power.draw", "power.limit", "utilization.gpu",
]

# The driver reports WHY clocks are reduced. Ask it, rather than inferring from
# the clock ratio: a healthy 3090 under full load boosts to roughly 1700 of its
# 2100 MHz maximum, so a ratio test calls every single run throttled. Newer
# drivers name these clocks_event_reasons.*, older ones clocks_throttle_reasons.*.
_REASONS = ["hw_thermal_slowdown", "sw_thermal_slowdown",
            "sw_power_cap", "hw_power_brake_slowdown"]
_REASON_PREFIXES = ("clocks_event_reasons", "clocks_throttle_reasons")


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def _query(fields: list[str]) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return None
    return [v.strip() for v in out.split(",")]


def probe() -> dict:
    if not available():
        return {"available": False, **{f: None for f in _FIELDS}}
    vals = _query(_FIELDS)
    if vals is None:
        return {"available": False, "error": "nvidia-smi query failed"}

    rec: dict = {"available": True}
    for f, v in zip(_FIELDS, vals):
        try:
            rec[f] = float(v) if f != "name" else v
        except ValueError:
            rec[f] = None
    if rec.get("clocks.sm") and rec.get("clocks.max.sm"):
        rec["clock_ratio"] = rec["clocks.sm"] / rec["clocks.max.sm"]

    # Best effort: a driver without these fields fails the whole query, so it is
    # asked for separately and its absence recorded rather than raised.
    for prefix in _REASON_PREFIXES:
        vals = _query([f"{prefix}.{r}" for r in _REASONS])
        if vals is not None and len(vals) == len(_REASONS):
            for r, v in zip(_REASONS, vals):
                rec[r] = (v.lower() == "active")
            rec["reasons_source"] = prefix
            break
    else:
        rec["reasons_source"] = None
    return rec


class Sampler:
    """Polls the GPU while a run is in flight.

    Start/end probes cannot tell thermal throttling from ordinary idle
    downclocking: an idle 3090 sits near 200 MHz against a 2100 MHz maximum, which
    looks identical to a badly throttled one. Worse, the end probe is usually taken
    after the load has already dropped, so it describes the recovery rather than the
    run. The only honest check samples during the work and judges only those samples
    where the GPU was actually busy.

        with gpu.Sampler() as s:
            ...run...
        s.summary()
    """

    def __init__(self, interval_s: float = 1.0, busy_util: float = 25.0) -> None:
        self.interval_s = interval_s
        self.busy_util = busy_util
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Sampler":
        if available():
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s * 3)

    def _loop(self) -> None:
        while not self._stop.is_set():
            rec = probe()
            if rec.get("available"):
                rec["t"] = time.time()
                self.samples.append(rec)
            self._stop.wait(self.interval_s)

    def summary(self, floor: float = 0.90) -> dict:
        if not self.samples:
            return {"checked": False, "reason": "no nvidia-smi", "n_samples": 0,
                    "throttled": None, "temp_max": None, "clock_ratio_p50": None,
                    "sm_clock_min": None, "sm_clock_max": None, "util_mean": None}

        busy = [s for s in self.samples
                if (s.get("utilization.gpu") or 0) >= self.busy_util]
        util = [s.get("utilization.gpu") or 0 for s in self.samples]
        temps = [s["temperature.gpu"] for s in self.samples if s.get("temperature.gpu")]

        base = {
            "n_samples": len(self.samples),
            "n_busy_samples": len(busy),
            "util_mean": sum(util) / len(util),
            "temp_max": max(temps) if temps else None,
            "temp_mean": (sum(temps) / len(temps)) if temps else None,
        }

        if not busy:
            # Nothing to judge: the GPU was never loaded during this run. Saying
            # "not throttled" here would be as wrong as saying "throttled".
            return {**base, "checked": False, "throttled": None,
                    "reason": f"GPU never exceeded {self.busy_util:.0f}% utilisation "
                              f"during this run; no throttling judgement possible",
                    "clock_ratio_p50": None, "sm_clock_min": None, "sm_clock_max": None}

        ratios = sorted(s["clock_ratio"] for s in busy if s.get("clock_ratio"))
        clocks = [s["clocks.sm"] for s in busy if s.get("clocks.sm")]
        p50 = ratios[len(ratios) // 2] if ratios else None

        # Authoritative: the driver says why clocks dropped. The clock ratio is kept
        # as context but is NOT the verdict -- under full load this card sits near
        # 1700/2100 MHz with nothing wrong, which a ratio test reads as throttling.
        thermal = any(s.get("hw_thermal_slowdown") or s.get("sw_thermal_slowdown")
                      for s in busy)
        power = any(s.get("sw_power_cap") or s.get("hw_power_brake_slowdown")
                    for s in busy)
        have_reasons = any(s.get("reasons_source") for s in busy)
        powers = [s["power.draw"] for s in busy if s.get("power.draw")]

        return {
            **base,
            "checked": True,
            "clock_ratio_p50": p50,
            "sm_clock_min": min(clocks) if clocks else None,
            "sm_clock_max": max(clocks) if clocks else None,
            "power_draw_max": max(powers) if powers else None,
            "power_limit": busy[0].get("power.limit"),
            "throttled": (thermal or power) if have_reasons else None,
            "thermal_throttled": thermal if have_reasons else None,
            "power_capped": power if have_reasons else None,
            "reason": None if have_reasons else
                      "driver exposes no clock-event reasons; clock ratio alone "
                      "cannot distinguish throttling from normal boost behaviour",
        }


def throttle_verdict(before: dict, after: dict, floor: float = 0.90) -> dict:
    """Flag a run whose SM clock sagged. A 350W 3090 throttles under sustained load;
    across a long sweep that turns into drift you will read as effect."""
    if not (before.get("available") and after.get("available")):
        # Same keys either way: a consumer must never KeyError just because a run
        # happened somewhere without a GPU.
        return {"checked": False, "reason": "no nvidia-smi", "throttled": None,
                "sm_clock_start": None, "sm_clock_end": None,
                "temp_start": None, "temp_end": None, "clock_ratio_end": None}
    r = after.get("clock_ratio")
    return {
        "checked": True,
        "sm_clock_start": before.get("clocks.sm"),
        "sm_clock_end": after.get("clocks.sm"),
        "temp_start": before.get("temperature.gpu"),
        "temp_end": after.get("temperature.gpu"),
        "clock_ratio_end": r,
        "throttled": (r is not None and r < floor),
    }
