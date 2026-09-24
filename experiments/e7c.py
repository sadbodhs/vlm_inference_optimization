#!/usr/bin/env python3
"""E7c: the cascade, industry-shaped -- DeepStream + tracker gating a separate VLM.

PLAN.md 12. The CV side is one DeepStream pipeline for all cameras, the way NVIDIA's
VSS blueprint builds it: NVDEC decode -> nvstreammux (batch N) -> nvinfer (YOLOv8s @
1280, FP16, every 6th frame) -> nvtracker (NvDCF) on every frame. The gate fires on
*track* events. The VLM side is the unchanged vLLM service from E7b.

Modes:
  offline   run the pipeline flat out over all 24 E7 clips, log tracks per 5 fps frame
  gate      replay track gates over those logs against MEVA's labels (as E7 did)
  prep      cut keyframe-aligned, looped camera files for the live run's offsets
  live      N real-time cameras -> DeepStream -> track gate -> vLLM; E7b's bar

The pipeline always runs in its own process (spawn): E7b showed that a detector
sharing a Python process with the VLM client inherits the client's GIL stalls.

    docker/run_ds.sh python3 experiments/e7c.py offline
    docker/run_harness.sh python3 experiments/e7c.py gate
    docker/run_ds.sh python3 experiments/e7c.py live --gate person-track-motion --arm roi --cams 8,10,12
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import collections
import gzip
import json
import multiprocessing as mp
import queue
import random
import statistics as st
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

WINDOW, STRIDE, FPS = 60, 6, 30
VLM_OFFSETS = (18, 48)
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON = {"person"}
VEHICLE = {"car", "truck", "bus", "motorcycle", "bicycle"}
MOVE_FRAC = 0.1
INFER_CFG = "/work/configs/e7c/config_infer_yolov8s_1280.txt"
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_CFG = ("/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/"
               "config_tracker_NvDCF_perf.yml")
FRESH_S, LATE_LIMIT_S = 2.0, 1.0
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"      # E7c; E7d passes --arm-file
TRACK_GATES = ("track-presence", "track-motion", "person-track-motion")


# ── the DeepStream process ───────────────────────────────────────────────────
def ds_worker(uris: list[str], batch: int, sync: bool, out_q, stop_evt):
    """Build and run the pipeline; emit one record per source per 5 fps frame:
    ("f", source, frame_num, pts_ns, t_probe_wall, [[cls, track_id, conf, x1, y1, x2, y2]])."""
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import GLib, Gst
    import pyds

    Gst.init(None)
    # The whole graph from a launch string, as the gst-launch smoke test that worked:
    # "uridecodebin ! m.sink_i" makes GStreamer link each decoder's dynamic pad in C.
    # Two Python versions -- linking in a pad-added callback, then requesting the
    # muxer pads up front -- both deadlocked intermittently at startup (0% GPU, no
    # frame ever), because any Python callback on N decoder threads during the
    # state change contends for the GIL with GStreamer's own locks. After this,
    # the only Python in the streaming path is the tracker probe, which runs only
    # once frames flow.
    # Explicit decode chains, not uridecodebin. Isolated with bare gst-launch: 8
    # uridecodebin sources hung at startup even with no inference or tracker
    # (decode + mux alone), with or without a queue per source and with the new
    # nvstreammux; the explicit filesrc ! demux ! h264parse ! nvv4l2decoder chain
    # completed every time. 4 or fewer uridecodebins happened to work.
    def chain(i, u):
        path = u[len("file://"):]
        demux = "matroskademux" if path.endswith(".mkv") else "avidemux"
        return (f"filesrc name=src{i} location={path} ! {demux} ! h264parse "
                f"! nvv4l2decoder ! m.sink_{i}")
    srcs = " ".join(chain(i, u) for i, u in enumerate(uris))
    desc = (f"nvstreammux name=m batch-size={batch} width=1920 height=1080 "
            f"batched-push-timeout=40000 live-source=0 "
            f"! nvinfer name=pgie config-file-path={INFER_CFG} "
            f"! nvtracker name=tracker ll-lib-file={TRACKER_LIB} ll-config-file={TRACKER_CFG} "
            f"tracker-width=960 tracker-height=544 gpu-id=0 "
            f"! fakesink name=sink sync={'true' if sync else 'false'} qos=false {srcs}")
    pipe = Gst.parse_launch(desc)
    trk = pipe.get_by_name("tracker")

    def count_hw_decoders(bin_):
        n, it = 0, bin_.iterate_recurse()
        while True:
            ok, el = it.next()
            if ok != Gst.IteratorResult.OK:
                break
            if el.get_factory() and el.get_factory().get_name() == "nvv4l2decoder":
                n += 1
        return n

    def probe(_pad, info):
        buf = info.get_buffer()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        t_now = time.time()
        lf = batch_meta.frame_meta_list
        while lf is not None:
            fm = pyds.NvDsFrameMeta.cast(lf.data)
            if fm.frame_num % STRIDE == 0:
                objs, lo = [], fm.obj_meta_list
                while lo is not None:
                    om = pyds.NvDsObjectMeta.cast(lo.data)
                    if om.class_id in COCO:
                        r = om.rect_params
                        objs.append([COCO[om.class_id], int(om.object_id), round(float(om.confidence), 3),
                                     round(r.left, 1), round(r.top, 1),
                                     round(r.left + r.width, 1), round(r.top + r.height, 1)])
                    lo = lo.next
                out_q.put(("f", fm.pad_index, fm.frame_num, fm.buf_pts, t_now, objs))
            lf = lf.next
        return Gst.PadProbeReturn.OK

    trk.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe)

    loop = GLib.MainLoop()

    started = {"done": False}

    def on_msg(_bus, msg):
        if (msg.type == Gst.MessageType.STATE_CHANGED and msg.src == pipe
                and not started["done"]):
            _old, new_state, _pending = msg.parse_state_changed()
            if new_state == Gst.State.PLAYING:
                started["done"] = True
                clock = pipe.get_clock()
                wall0 = time.time() - (clock.get_time() - pipe.get_base_time()) / 1e9
                hw = count_hw_decoders(pipe)
                out_q.put(("start", wall0, list(range(hw, len(uris))) if hw < len(uris) else []))
        elif msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            out_q.put(("error", f"{err}: {dbg}"))
            loop.quit()
        return True

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_msg)

    # Never block the main thread on get_state(): the decoders' pad-added callbacks
    # are Python and run on GStreamer's threads, so a main thread parked in a C call
    # that holds the GIL deadlocks the pipeline before its first frame. That is how
    # the first offline run sat at 0% GPU for two hours. The clock origin is taken
    # from the PLAYING state-change message instead.
    pipe.set_state(Gst.State.PLAYING)

    def check_stop():
        if stop_evt.is_set():
            pipe.send_event(Gst.Event.new_eos())
            return False
        return True
    GLib.timeout_add(200, check_stop)
    loop.run()
    pipe.set_state(Gst.State.NULL)
    out_q.put(("eos",))


def start_ds(uris, batch, sync):
    ctx = mp.get_context("spawn")
    q, stop = ctx.Queue(maxsize=0), ctx.Event()
    p = ctx.Process(target=ds_worker, args=(uris, batch, sync, q, stop), daemon=True)
    p.start()
    return p, q, stop


# ── gates on tracks ──────────────────────────────────────────────────────────
def _by_id(objs, groups):
    return {o[1]: o for o in objs if o[0] in groups}


def track_changed(prev: list, cur: list, groups: set) -> bool:
    """A track was born, was lost, or moved > MOVE_FRAC of its height. IDs come from
    the tracker, so a box that flickers is still one target -- the thing E7's
    box-matching gate could not know."""
    a, b = _by_id(prev, groups), _by_id(cur, groups)
    if set(a) != set(b):
        return True
    for tid, oa in a.items():
        ob = b[tid]
        h = max(oa[6] - oa[4], ob[6] - ob[4], 1.0)
        dx = (oa[3] + oa[5]) / 2 - (ob[3] + ob[5]) / 2
        dy = (oa[4] + oa[6]) / 2 - (ob[4] + ob[6]) / 2
        if (dx * dx + dy * dy) ** 0.5 > MOVE_FRAC * h:
            return True
    return False


def gate_window(gate: str, win: list[list], prev: list) -> bool:
    if gate == "dense":
        return True
    if gate == "track-presence":
        return any(o[0] in PERSON | VEHICLE for f in win for o in f)
    groups = PERSON | VEHICLE if gate == "track-motion" else PERSON
    seq = [prev] + win
    return any(track_changed(x, y, groups) for x, y in zip(seq, seq[1:]))


# ── offline: tracks for every clip ───────────────────────────────────────────
def cmd_offline(args):
    sel = json.load(open(args.selection))["clips"]
    out = Path(args.track_dir); out.mkdir(parents=True, exist_ok=True)
    todo = [c["clip"] for c in sel if not (out / f"{c['clip']}.jsonl.gz").exists()]
    for g in range(0, len(todo), args.group):
        clips = todo[g:g + args.group]
        uris = [f"file://{Path(args.video_dir).resolve()}/{c}.r13.avi" for c in clips]
        t0 = time.time()
        p, q, _ = start_ds(uris, len(uris), sync=False)
        recs = collections.defaultdict(list)
        while True:
            m = q.get()
            if m[0] == "f":
                recs[m[1]].append({"f": m[2], "o": m[5]})
            elif m[0] == "start":
                if m[2]:
                    sys.exit(f"sources {m[2]} were not NVMM-decoded -- not the pipeline claimed")
            elif m[0] == "error":
                sys.exit(f"pipeline error: {m[1]}")
            elif m[0] == "eos":
                break
        p.join(timeout=30)
        for i, c in enumerate(clips):
            rows = sorted(recs[i], key=lambda r: r["f"])
            with gzip.open(out / f"{c}.jsonl.gz", "wt") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n")
            print(f"  {c}: {len(rows)} sampled frames", flush=True)
            if len(rows) < 1450:
                sys.exit(f"{c}: only {len(rows)} frames tracked")
        n = sum(len(v) for v in recs.values()) * STRIDE
        print(f"  group of {len(clips)}: {n} frames in {time.time()-t0:.0f}s "
              f"({n/(time.time()-t0):.0f} fps incl. startup)", flush=True)



# ── parity: does DeepStream see what E7's detector saw? ──────────────────────
def cmd_parity(args):
    """Before any gate comparison: the DeepStream path (nvinfer + NvDCF) against the
    ultralytics path E7 used, on the same frames and against MEVA's actors. A gate
    that fires less because its detector sees less is not a better gate."""
    from e7_gate import HEIGHT_BINS, hbin, iou, load_geom, load_types
    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    root = Path("data/meva/meva-data-repo")
    tot = collections.Counter()
    act = collections.defaultdict(lambda: [0, 0, 0])      # hbin -> [n, ds hit, yolo hit]
    for c in sel:
        clip = c["clip"]
        ds = {json.loads(l)["f"]: json.loads(l)["o"]
              for l in gzip.open(Path(args.track_dir) / f"{clip}.jsonl.gz", "rt") if l.strip()}
        yo = {json.loads(l)["f"]: json.loads(l)["d"]
              for l in gzip.open(f"data/meva/det/{clip}.1280.jsonl.gz", "rt") if l.strip()}
        for f in set(ds) & set(yo):
            d_ = [o for o in ds[f]]
            y_ = [d for d in yo[f]]
            tot["ds"] += len(d_); tot["yolo"] += len(y_)
            used = set()
            for y in y_:
                grp = PERSON if y[0] in PERSON else VEHICLE
                best = max(((iou(y[2:], o[3:]), j) for j, o in enumerate(d_)
                            if j not in used and o[0] in grp), default=(0, -1))
                if best[0] >= 0.5:
                    used.add(best[1]); tot["yolo_matched"] += 1
            tot[f"bin_{c['bin']}_ds"] += len(d_); tot[f"bin_{c['bin']}_yolo"] += len(y_)
        meta = idx[clip]
        ann = root / meta["annotation_dir"]
        tracks, types = load_geom(ann / f"{clip}.geom.yml"), load_types(ann / f"{clip}.types.yml")
        for a in meta["activities"]:
            for tid in a["actors"]:
                if types.get(tid) != "person":
                    continue
                for fr, box in tracks.get(tid, {}).items():
                    if fr % STRIDE or not (a["start"] <= fr <= a["end"]) or fr not in ds:
                        continue
                    k = hbin(box[3] - box[1])
                    act[k][0] += 1
                    act[k][1] += any(o[0] == "person" and iou(o[3:], box) >= 0.3 for o in ds[fr])
                    act[k][2] += any(d[0] == "person" and iou(d[2:], box) >= 0.3 for d in yo.get(fr, []))
    print(f"objects on shared frames: DeepStream {tot['ds']}, ultralytics {tot['yolo']}; "
          f"ultralytics boxes matched by a DeepStream box (IoU>=0.5): "
          f"{100*tot['yolo_matched']/max(tot['yolo'],1):.1f}%")
    for b in ("empty", "sparse", "moderate", "busy"):
        print(f"  {b:9s} DeepStream {tot[f'bin_{b}_ds']:7d}   ultralytics {tot[f'bin_{b}_yolo']:7d}")
    print("annotated people found, by height (DeepStream / ultralytics):")
    rows = {}
    for lo, hi in HEIGHT_BINS:
        k = f"{lo}-{hi}" if hi < 10_000 else f"{lo}+"
        n, h1, h2 = act[k]
        rows[k] = {"n": n, "deepstream": h1 / n if n else None, "ultralytics": h2 / n if n else None}
        if n:
            print(f"  {k:>8s} n={n:6d}   {100*h1/n:5.1f}%   {100*h2/n:5.1f}%")
    Path("results/e7c").mkdir(parents=True, exist_ok=True)
    json.dump({"objects": dict(tot), "actor_recall_by_height": rows},
              open("results/e7c/parity.json", "w"), indent=1)

# ── gate: price track gates against MEVA, exactly as E7 priced box gates ─────
def cmd_gate(args):
    from e7_gate import GATES as BOX_GATES
    sel = json.load(open(args.selection))["clips"]
    idx = {c["clip"]: c for c in json.load(open(args.index))["clips"]}
    calls = collections.defaultdict(lambda: [0, 0])
    rec = collections.defaultdict(lambda: [0, 0])
    for c in sel:
        clip, b, meta = c["clip"], c["bin"], idx[c["clip"]]
        frames = {}
        with gzip.open(Path(args.track_dir) / f"{clip}.jsonl.gz", "rt") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line); frames[r["f"]] = r["o"]
        n_win = meta["n_frames"] // WINDOW
        for g in TRACK_GATES:
            fired = []
            for w in range(n_win):
                win = [frames.get(fr, []) for fr in range(w * WINDOW, (w + 1) * WINDOW, STRIDE)]
                fired.append(gate_window(g, win, frames.get(w * WINDOW - STRIDE, [])))
            for key in ((g, b), (g, "all")):
                calls[key][0] += sum(fired); calls[key][1] += len(fired)
            for a in meta["activities"]:
                ws = range(a["start"] // WINDOW, min(n_win - 1, a["end"] // WINDOW) + 1)
                hit = any(fired[w] for w in ws)
                for key in ((g, b), (g, "all")):
                    rec[key][0] += hit; rec[key][1] += 1
    e7 = json.load(open(args.e7_gates))
    bins = ["empty", "sparse", "moderate", "busy", "all"]
    rows = []
    print(f"{'gate':22s}" + "".join(f"{b:>18s}" for b in bins))
    for g in TRACK_GATES:
        cells = []
        for b in bins:
            cr, rc = calls[(g, b)], rec[(g, b)]
            rows.append({"gate": g, "bin": b, "call_rate": cr[0] / cr[1] if cr[1] else None,
                         "recall": rc[0] / rc[1] if rc[1] else None, "instances": rc[1]})
            cells.append((f"{100*cr[0]/cr[1]:5.1f}% / " + (f"{100*rc[0]/rc[1]:5.1f}%" if rc[1] else "  -  ")).rjust(18))
        print(f"{g:22s}" + "".join(cells))
    print("--- E7 box gates, YOLO @ 1280, for comparison ---")
    for g in ("presence", "motion", "person-motion"):
        cells = []
        for b in bins:
            c_ = e7["calls"].get(f"1280|{g}|{b}"); r_ = e7["recall"].get(f"1280|{g}|{b}")
            cells.append((f"{100*c_['rate']:5.1f}% / " + (f"{100*r_['rate']:5.1f}%" if r_ and r_["rate"] is not None else "  -  ")).rjust(18))
        print(f"{g:22s}" + "".join(cells))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"meta": {"tracker": "NvDCF perf", "detector": "YOLOv8s @1280 FP16 interval=5",
                        "move_frac": MOVE_FRAC}, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


# ── prep: keyframe-aligned camera files for the live offsets ─────────────────
def live_cameras(n: int, clips_interleaved: list[dict]) -> list[tuple[dict, int]]:
    """Exactly E7b's assignment: camera i -> interleaved clip i, start offset drawn
    from Random(1000 + n) in camera order, a multiple of 60 (a keyframe in MEVA)."""
    rng = random.Random(1000 + n)
    return [(clips_interleaved[i % len(clips_interleaved)], rng.randrange(0, 9000 // WINDOW - 10) * WINDOW)
            for i in range(n)]


def interleaved(selection: str) -> list[dict]:
    sel = json.load(open(selection))["clips"]
    bins = ["empty", "sparse", "moderate", "busy"]
    by = {b: [c for c in sel if c["bin"] == b] for b in bins}
    return [by[b][i] for i in range(6) for b in bins]


def cmd_prep(args):
    """Print "clip start" lines for every camera of every N requested; the shell
    driver cuts them with ffmpeg (stream copy from a keyframe, looped once)."""
    need = set()
    for n in [int(x) for x in args.cams.split(",")]:
        for c, k in live_cameras(n, interleaved(args.selection)):
            need.add((c["clip"], k))
    for clip, k in sorted(need):
        print(clip, k)


# ── live ─────────────────────────────────────────────────────────────────────
class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.util, self.mem, self.stop = [], [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True,
                                   text=True, timeout=5).stdout.strip().split(",")
                self.util.append(float(o[0])); self.mem.append(float(o[1]))
            except Exception:
                pass
            self.stop.wait(1.0)


async def live_once(args, n: int) -> tuple[dict, list]:
    import httpx
    from bench.imaging import resize_to_budget
    from e7_vlm import QUESTION, crop_jpeg, roi_box

    cams = live_cameras(n, interleaved(args.selection))
    # real frame counts (ffprobe, written by scripts/e7c_prep.sh): clips run 9000-9018
    # frames, and the camera files wrap at the real end, so the modulus must be exact
    nframes = json.loads((Path(args.trim_dir) / "nframes.json").read_text())
    uris = [f"file://{Path(args.trim_dir).resolve()}/{c['clip']}_{k}.mkv" for c, k in cams]
    for u in uris:
        if not Path(u[7:]).exists():
            sys.exit(f"missing camera file {u} -- run prep first")

    loop = asyncio.get_running_loop()
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=8)
    http = httpx.AsyncClient(limits=httpx.Limits(max_connections=512), timeout=120)
    state = [{"frames": {}, "tcap": {}} for _ in cams]
    results, pending, late = [], set(), []
    counters = {"windows": 0, "fired": 0}
    t = {"wall0": None}

    async def send(ci, w_abs, t_newest, boxes_pair):
        clip = cams[ci][0]["clip"]
        raws = [(Path(args.frame_dir) / clip / f"{w_abs*WINDOW+o:05d}.jpg").read_bytes()
                for o in VLM_OFFSETS]

        def build():
            imgs = raws
            if args.arm == "roi":
                box = roi_box(boxes_pair[0], boxes_pair[1], 1920, 1080)
                if box is not None:
                    imgs = [crop_jpeg(r, box) for r in raws]
            return [base64.b64encode(resize_to_budget(r, args.max_pixels)[0]).decode() for r in imgs]
        b64 = await loop.run_in_executor(pool, build)
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}
                   for x in b64] + [{"type": "text", "text": QUESTION}]
        body = {"model": args.model, "messages": [{"role": "user", "content": content}],
                "max_tokens": 16, "temperature": 0, "stream": True, **args.extra_body}
        t_send, t_first, ok = time.time(), None, False
        try:
            async with http.stream("POST", f"{args.base_url}/v1/chat/completions", json=body) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]" and t_first is None:
                        if json.loads(line[6:])["choices"][0]["delta"].get("content"):
                            t_first = time.time()
                ok = r.status_code == 200
        except Exception:
            ok = False
        results.append({"cam": clip, "w": w_abs, "ok": ok, "staleness_s": time.time() - t_newest,
                        "ttft_s": (t_first or time.time()) - t_send, "measured": True})

    def on_frame(src, fn, pts, t_probe, objs):
        c, k = cams[src]
        nf = nframes[c["clip"]]
        a = (k + fn) % nf                       # absolute frame in the source clip
        t_cap = t["wall0"] + pts / 1e9
        st_ = state[src]
        st_["frames"][a] = objs
        st_["tcap"][a] = t_cap
        in_window = args.warmup <= pts / 1e9 <= args.warmup + args.duration
        if in_window:
            late.append(t_probe - t_cap)
        if a % WINDOW == WINDOW - STRIDE:        # last sampled frame of its window
            w = a // WINDOW
            base = w * WINDOW
            win = [st_["frames"].get(f, []) for f in range(base, base + WINDOW, STRIDE)]
            prev = st_["frames"].get(base - STRIDE, [])
            if in_window:
                counters["windows"] += 1
            if gate_window(args.gate, win, prev):
                fr = [base + o for o in VLM_OFFSETS]
                if in_window and all(f in st_["tcap"] for f in fr):
                    counters["fired"] += 1
                    pairs = [[[o[0], o[2], *o[3:7]] for o in st_["frames"].get(f, [])] for f in fr]
                    tk = asyncio.ensure_future(send(src, w, st_["tcap"][fr[1]], pairs))
                    pending.add(tk); tk.add_done_callback(pending.discard)
            for f in [f for f in st_["frames"] if f < a - 2 * WINDOW or f > a]:
                st_["frames"].pop(f, None); st_["tcap"].pop(f, None)

    proc, q, stop = start_ds(uris, n, sync=True)
    sampler = GpuSampler(); sampler.start()
    done = asyncio.Event()

    def relay():
        held = []          # frames that preroll before PLAYING wait for the clock origin
        while True:
            m = q.get()
            if m[0] == "start":
                t["wall0"] = m[1]
                if m[2]:
                    print(f"  WARNING: sources {m[2]} not NVMM-decoded", flush=True)
                for h in held:
                    loop.call_soon_threadsafe(on_frame, *h[1:])
                held.clear()
            elif m[0] == "f":
                if t["wall0"] is None:
                    held.append(m)
                else:
                    loop.call_soon_threadsafe(on_frame, *m[1:])
            elif m[0] in ("eos", "error"):
                if m[0] == "error":
                    print("  pipeline error:", m[1], flush=True)
                loop.call_soon_threadsafe(done.set)
                return
    threading.Thread(target=relay, daemon=True).start()

    # the camera files are longer than warm-up + duration; stop at the end of the window
    while t["wall0"] is None and not done.is_set():
        await asyncio.sleep(0.1)
    if t["wall0"] is not None:
        await asyncio.sleep(max(0.0, t["wall0"] + args.warmup + args.duration + 1 - time.time()))
    stop.set()
    await done.wait()
    if pending:
        await asyncio.wait(pending, timeout=30)
    sampler.stop.set()
    await http.aclose()
    proc.join(timeout=30)

    ok = [r for r in results if r["ok"]]
    stale = sorted(r["staleness_s"] for r in ok)
    lat = sorted(late)
    q_ = lambda v, p: v[min(len(v) - 1, int(p * len(v)))] if v else None
    fresh = sum(1 for s in stale if s <= FRESH_S) / len(results) if results else None
    row = {"cams": n, "gate": args.gate, "arm": args.arm, "detector": "deepstream",
           "windows": counters["windows"], "sent": len(results), "ok": len(ok),
           "call_rate": counters["fired"] / counters["windows"] if counters["windows"] else None,
           "staleness_p50_s": q_(stale, 0.5), "staleness_p95_s": q_(stale, 0.95),
           "fresh_fraction": fresh,
           "det_latency_p50_s": q_(lat, 0.5), "det_latency_p99_s": q_(lat, 0.99),
           "det_drop_frac": 0.0,
           "gpu_util_mean": st.mean(sampler.util) if sampler.util else None,
           "gpu_mem_max_mib": max(sampler.mem) if sampler.mem else None}
    row["supported"] = bool(fresh is not None and fresh >= 0.95
                            and (row["det_latency_p99_s"] or 0) < LATE_LIMIT_S)
    return row, results


def next_count(rows: dict, plan: list[int]) -> int | None:
    """E7d's search: walk the planned counts until two consecutive failures, then
    fill the gap between the highest pass and the lowest failure above it; if even
    the first count fails, step down until one passes."""
    fails = 0
    for n in plan:
        if n not in rows:
            return n
        fails = 0 if rows[n]["supported"] else fails + 1
        if fails >= 2:
            break
    ok = [n for n, r in rows.items() if r["supported"]]
    if not ok:
        lo = min(rows)
        return lo - 1 if lo > 1 else None
    best = max(ok)
    above = [n for n, r in rows.items() if not r["supported"] and n > best]
    if above and min(above) - best > 1:
        return best + 1
    return None


async def cmd_live(args):
    tag = f"{args.gate}-{args.arm}-deepstream"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, fails = [], 0
    plan = [int(x) for x in args.cams.split(",")]
    done_rows: dict[int, dict] = {}
    while True:
        if args.search:
            n = next_count(done_rows, plan)
            if n is None:
                break
        else:
            if not plan:
                break
            n = plan.pop(0)
        row, detail = await live_once(args, n)
        row["model"] = args.model
        done_rows[n] = row
        rows.append(row)
        (out / f"{tag}-n{n}.jsonl").write_text("\n".join(json.dumps(r) for r in detail) + "\n")
        print(f"  N={n:<3} calls {100*(row['call_rate'] or 0):5.1f}%  sent {row['sent']:4d}  "
              f"fresh {100*(row['fresh_fraction'] or 0):5.1f}%  stale p95 {row['staleness_p95_s'] or 0:5.2f}s  "
              f"det-path p99 {row['det_latency_p99_s'] or 0:5.3f}s  gpu {row['gpu_util_mean'] or 0:4.0f}%  "
              f"mem {row['gpu_mem_max_mib'] or 0:6.0f}  {'OK' if row['supported'] else 'no'}", flush=True)
        fails = 0 if row["supported"] else fails + 1
        if fails >= 2 and not args.search:
            break
        await asyncio.sleep(5)
    prev = out / f"{tag}.json"
    old = json.loads(prev.read_text())["rows"] if prev.exists() else []
    merged = {r["cams"]: r for r in old + rows}
    json.dump({"meta": {"experiment": "e7c-live", "gate": args.gate, "arm": args.arm,
                        "model": args.model, "vlm_arm": args.arm_id,
                        "duration_s": args.duration, "warmup_s": args.warmup, "fresh_s": FRESH_S,
                        "late_limit_s": LATE_LIMIT_S, "tracker": "NvDCF perf"},
               "rows": [merged[k] for k in sorted(merged)]}, open(prev, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["offline", "gate", "prep", "live", "parity"])
    ap.add_argument("--selection", default="results/e7/selection.json")
    ap.add_argument("--index", default="data/meva/index.json")
    ap.add_argument("--video-dir", default="data/meva/video")
    ap.add_argument("--frame-dir", default="data/meva/frames5")
    ap.add_argument("--track-dir", default="data/meva/tracks")
    ap.add_argument("--trim-dir", default="data/meva/trim")
    ap.add_argument("--e7-gates", default="results/e7/gates.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--gate", default="person-track-motion",
                    choices=["dense", "track-presence", "track-motion", "person-track-motion"])
    ap.add_argument("--arm", default="roi", choices=["full", "roi"])
    ap.add_argument("--cams", default="8,10,12")
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--max-pixels", type=int, default=451584)
    ap.add_argument("--base-url", default="http://vlm-server:8000")
    ap.add_argument("--arm-file", default=None,
                    help="VLM arm YAML (E7d); default: E7c's Qwen2.5-VL-7B-AWQ")
    ap.add_argument("--search", action="store_true",
                    help="step through --cams until two failures, then fill the gap")
    args = ap.parse_args()
    args.model, args.extra_body, args.arm_id = MODEL_ID, {}, "V_vllm_video"
    if args.arm_file:
        from bench.arms import Arm
        a = Arm.load(args.arm_file)
        args.model, args.extra_body, args.arm_id = a.model, dict(a.extra_body), a.id
    if args.mode == "offline":
        cmd_offline(args)
    elif args.mode == "gate":
        args.out = args.out or "results/e7c/gates.json"
        cmd_gate(args)
    elif args.mode == "prep":
        cmd_prep(args)
    elif args.mode == "parity":
        cmd_parity(args)
    else:
        args.out = args.out or "results/e7c"
        asyncio.run(cmd_live(args))


if __name__ == "__main__":
    main()
