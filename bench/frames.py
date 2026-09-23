"""Uniform frame sampling from a clip.

The frame count is E6's independent variable, so extraction happens per-arm rather
than being baked into the dataset. It runs before the request is issued -- decoding
a clip inside the timed path would bill video decode to the server's TTFT, the same
mistake as resizing inside the request loop.

ffmpeg is invoked once per clip and asked for exactly N evenly spaced frames, so
decode cost does not scale with how many frames are kept.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class FrameError(RuntimeError):
    pass


def duration_s(path: str | Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        return float(out)
    except Exception:  # noqa: BLE001
        return None


def sample_frames(path: str | Path, n: int, timeout_s: float = 120.0) -> list[bytes]:
    """`n` JPEG frames spread evenly across the clip, in temporal order.

    Order is preserved deliberately: every question about direction, speed or
    ordering depends on it, and shuffling frames would silently turn a temporal
    benchmark into a bag-of-frames one.
    """
    path = Path(path)
    if not path.exists():
        raise FrameError(f"missing clip: {path}")
    if n < 1:
        raise FrameError("n must be >= 1")

    dur = duration_s(path)
    out_dir = Path("/tmp") / f"frames-{path.stem}-{n}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if n == 1:
        # A single frame from the middle, not the first: opening frames are often
        # black or a title card, which would understate what one frame can do.
        ts = (dur or 2.0) / 2
        cmd = ["ffmpeg", "-v", "error", "-ss", f"{ts:.3f}", "-i", str(path),
               "-frames:v", "1", "-q:v", "3", str(out_dir / "f-001.jpg")]
    else:
        # fps filter chosen so the whole clip yields ~n frames, then capped.
        fps = n / dur if dur and dur > 0 else 2.0
        cmd = ["ffmpeg", "-v", "error", "-i", str(path),
               "-vf", f"fps={fps:.6f}", "-frames:v", str(n),
               "-q:v", "3", str(out_dir / "f-%03d.jpg")]

    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout_s, check=True)
    except subprocess.CalledProcessError as e:
        raise FrameError(f"ffmpeg failed on {path.name}: "
                         f"{e.stderr.decode('utf-8','replace')[:200]}") from e

    frames = sorted(out_dir.glob("f-*.jpg"))
    if not frames:
        raise FrameError(f"no frames extracted from {path.name}")

    # Short clips can yield fewer than asked; repeat the last rather than failing,
    # and the caller records how many were actually distinct.
    data = [f.read_bytes() for f in frames]
    while len(data) < n:
        data.append(data[-1])
    out = data[:n]
    shutil.rmtree(out_dir, ignore_errors=True)
    return out
