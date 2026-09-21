"""GPU state probe. Thermal throttling is on the list of things nobody benchmarks,
so temperature and clocks are recorded with every run rather than assumed stable.

Returns None-filled records off-GPU (e.g. the Mac Studio) instead of raising, so the
harness stays runnable for dry runs.
"""
from __future__ import annotations

import shutil
import subprocess

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


def throttle_verdict(before: dict, after: dict, floor: float = 0.90) -> dict:
    """Flag a run whose SM clock sagged. A 350W 3090 throttles under sustained load;
    across a long sweep that turns into drift you will read as effect."""
    if not (before.get("available") and after.get("available")):
        return {"checked": False, "reason": "no nvidia-smi"}
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
