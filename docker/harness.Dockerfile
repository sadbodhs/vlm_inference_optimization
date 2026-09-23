# Harness container. CPU only -- it is an HTTP client and a plotter, nothing more.
# Kept deliberately separate from any serving image so the measurement code is
# identical across every arm (R1). Swapping vLLM for SGLang must not change this.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/mpl

# ffmpeg for E6: clips are decoded to frames client-side, before the request is
# timed, so video decode is never billed to the server's TTFT.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Source is bind-mounted at run time, not baked in: results land on the host and
# an edit does not cost an image rebuild.
CMD ["python3", "-c", "print('harness image; mount the repo at /work and pass a command')"]
