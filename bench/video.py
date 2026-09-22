"""Reading frames from continuously-running RTSP decoders.

The decoder writes one file per stream, replaced in place. That is the right model
for live video -- a consumer should only ever see the newest frame -- but it means
a reader can catch a half-written file. JPEG makes that detectable: a complete file
ends with the EOI marker FFD9, so a torn read can be retried rather than silently
sent to the model as a corrupt image.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path

JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


@dataclass
class Frame:
    stream: str
    captured_s: float   # wall clock, from the file's mtime
    read_s: float       # when this process read it
    data: bytes

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode()

    @property
    def age_at_read_ms(self) -> float:
        return (self.read_s - self.captured_s) * 1e3


def read_latest(frame_dir: Path, stream: str, retries: int = 5,
                retry_delay_s: float = 0.01) -> Frame | None:
    """Newest frame for one stream, or None if nothing readable yet."""
    path = Path(frame_dir) / stream / "latest.jpg"
    for _ in range(retries):
        try:
            st = path.stat()
            data = path.read_bytes()
        except (FileNotFoundError, OSError):
            time.sleep(retry_delay_s)
            continue
        # Complete JPEG, not a file caught mid-write.
        if len(data) > 4 and data[:2] == JPEG_SOI and data[-2:] == JPEG_EOI:
            return Frame(stream=stream, captured_s=st.st_mtime,
                         read_s=time.time(), data=data)
        time.sleep(retry_delay_s)
    return None


def wait_for_streams(frame_dir: Path, streams: list[str],
                     timeout_s: float = 60.0) -> list[str]:
    """Block until each stream has produced a readable frame. Returns the live ones.

    A decoder that never produces a frame would otherwise show up as excellent
    staleness numbers for a stream nobody is actually watching.
    """
    deadline = time.time() + timeout_s
    ready: set[str] = set()
    while time.time() < deadline and len(ready) < len(streams):
        for s in streams:
            if s not in ready and read_latest(frame_dir, s, retries=1) is not None:
                ready.add(s)
        if len(ready) < len(streams):
            time.sleep(0.5)
    return [s for s in streams if s in ready]
