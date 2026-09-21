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
    "clocks.sm", "clocks.max.sm", "power.draw", "utilization.gpu",
]


def available() -> bool:
    return shutil.which("nvidia-smi") is not None


def probe() -> dict:
    if not available():
        return {"available": False, **{f: None for f in _FIELDS}}
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(_FIELDS)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}

    vals = [v.strip() for v in out.split(",")]
    rec: dict = {"available": True}
    for f, v in zip(_FIELDS, vals):
        try:
            rec[f] = float(v) if f != "name" else v
        except ValueError:
            rec[f] = None
    if rec.get("clocks.sm") and rec.get("clocks.max.sm"):
        rec["clock_ratio"] = rec["clocks.sm"] / rec["clocks.max.sm"]
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
        return {
            **base,
            "checked": True,
            "clock_ratio_p50": p50,
            "sm_clock_min": min(clocks) if clocks else None,
            "sm_clock_max": max(clocks) if clocks else None,
            "throttled": (p50 is not None and p50 < floor),
            "reason": None,
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
