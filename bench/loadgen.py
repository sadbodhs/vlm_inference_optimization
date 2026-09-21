"""Load generators.

Open loop (Poisson arrivals) is the default and the only one whose throughput
numbers mean anything. A closed loop -- fire N, wait, fire N again -- cannot
produce a queue, so it cannot show you queueing delay, and it silently rate-limits
itself to whatever the server can do. That makes every stack look well-behaved.

Closed loop is kept only for concurrency=1 latency probes (E1), where there is no
queue to model anyway.
"""
from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable

from .result import RequestResult

SendFn = Callable[[int, float], Awaitable[RequestResult]]
"""(index, scheduled_s) -> RequestResult"""


async def open_loop(
    n: int, rate_qps: float, send: SendFn, *, seed: int = 0
) -> list[RequestResult]:
    """Poisson arrivals at `rate_qps`, independent of how fast the server replies."""
    rng = random.Random(seed)
    origin = time.perf_counter()

    offsets: list[float] = []
    t = 0.0
    for _ in range(n):
        offsets.append(t)
        t += rng.expovariate(rate_qps)

    async def one(i: int, offset: float) -> RequestResult:
        scheduled = origin + offset
        delay = scheduled - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        return await send(i, scheduled)

    return list(await asyncio.gather(*(one(i, o) for i, o in enumerate(offsets))))


async def closed_loop(
    n: int, concurrency: int, send: SendFn
) -> list[RequestResult]:
    """`concurrency` workers each pulling from a shared queue. Latency probes only."""
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(n):
        queue.put_nowait(i)
    out: list[RequestResult] = []

    async def worker() -> None:
        while True:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            # scheduled == sent by construction: a closed loop has no arrival process,
            # so loadgen_lag is meaningless here and is reported as ~0 on purpose.
            out.append(await send(i, time.perf_counter()))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return out


async def sequential(n: int, send: SendFn) -> list[RequestResult]:
    """Strictly one at a time. The only honest way to measure single-stream decode."""
    return await closed_loop(n, 1, send)
