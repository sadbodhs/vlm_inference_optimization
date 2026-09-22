"""Exact weight split from safetensors headers -- no tensor data is read.

The decode roofline must count only what decode re-reads: the LLM weights. The
vision tower runs once in prefill. Estimating that split is how E1 first produced
an impossible 114%-of-roofline reading, so it is measured here instead.
"""
import glob, json, struct, sys, collections

DT = {"F32":4,"F16":2,"BF16":2,"I8":1,"U8":1,"I32":4,"I64":8,"F8_E4M3":1,"F8_E5M2":1,"BOOL":1}

def split(pattern):
    groups = collections.Counter()
    for f in sorted(glob.glob(pattern)):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            nel = 1
            for d in meta["shape"]:
                nel *= d
            nbytes = nel * DT.get(meta["dtype"], 2)
            key = "vision" if (".visual." in name or name.startswith("visual.")) else "llm"
            groups[key] += nbytes
    return groups

for label, pat in (
    ("Qwen2.5-VL-7B-AWQ", "/home/suchi/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct-AWQ/snapshots/*/*.safetensors"),
    ("Qwen3-VL-8B-AWQ",   "/home/suchi/.cache/huggingface/hub/models--cyankiwi--Qwen3-VL-8B-Instruct-AWQ-4bit/snapshots/*/*.safetensors"),
):
    g = split(pat)
    tot = sum(g.values())
    if not tot:
        print(f"{label}: no weights found"); continue
    print(f"{label}")
    print(f"   vision tower : {g['vision']/1e9:6.3f} GB   (runs once in prefill)")
    print(f"   LLM (decode) : {g['llm']/1e9:6.3f} GB   <- roofline basis")
    print(f"   total        : {tot/1e9:6.3f} GB")
    print(f"   roofline     : {936e9/g['llm']:6.1f} tok/s")
