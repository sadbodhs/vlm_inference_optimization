"""Streaming client for OpenAI-compatible VLM servers (vLLM, SGLang, TRT-LLM, ...).

Every arm speaks this one protocol. That is what makes R1 enforceable: swapping
the serving stack changes a URL, not the measurement path.
"""
from __future__ import annotations

import json
import time

import httpx

from .result import RequestResult


class StreamError(Exception):
    pass


def _origin() -> float:
    return time.perf_counter()


async def stream_chat(
    http: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    arm_id: str,
    sample_id: str,
    scheduled_s: float,
    max_tokens: int = 128,
    temperature: float = 0.0,
    extra_body: dict | None = None,
    timeout_s: float = 300.0,
) -> RequestResult:
    """Issue one streaming chat completion and time it token by token.

    TTFT is measured to the first chunk carrying *non-empty content*. Servers emit
    a role-only opening delta; counting that as the first token understates TTFT by
    the width of one network hop and is a common way to accidentally win a benchmark.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # vLLM/SGLang return a final usage chunk when asked; this is how we get
        # authoritative prompt_tokens rather than guessing with a local tokenizer.
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        payload.update(extra_body)

    res = RequestResult(
        arm_id=arm_id, sample_id=sample_id, scheduled_s=scheduled_s, sent_s=_origin()
    )
    chunks: list[str] = []
    prev_token_s: float | None = None

    try:
        async with http.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            timeout=timeout_s,
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise StreamError(f"HTTP {resp.status_code}: {body}")

            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break

                now = _origin()
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if usage := obj.get("usage"):
                    res.prompt_tokens = usage.get("prompt_tokens")
                    res.completion_tokens = usage.get("completion_tokens")

                for choice in obj.get("choices") or []:
                    content = (choice.get("delta") or {}).get("content")
                    if not content:
                        continue
                    chunks.append(content)
                    if res.first_token_s is None:
                        res.first_token_s = now
                    else:
                        res.itl_ms.append((now - prev_token_s) * 1e3)
                    prev_token_s = now
                    res.last_token_s = now

        res.text = "".join(chunks)
        if res.first_token_s is None:
            res.ok = False
            res.error = "no content tokens received"
    except Exception as exc:  # noqa: BLE001 - a failed request is data, not a crash
        res.ok = False
        res.error = f"{type(exc).__name__}: {exc}"

    return res


async def wait_for_server(base_url: str, timeout_s: float = 600.0) -> bool:
    """Block until the server answers /v1/models. Returns False on timeout.

    Cold start is one of the things nobody benchmarks, so the caller is expected to
    time this and record it rather than hiding it in setup.
    """
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient() as http:
        while time.time() < deadline:
            try:
                r = await http.get(f"{base_url.rstrip('/')}/v1/models", timeout=5.0)
                if r.status_code == 200:
                    return True
            except Exception:  # noqa: BLE001
                pass
            import asyncio

            await asyncio.sleep(1.0)
    return False
