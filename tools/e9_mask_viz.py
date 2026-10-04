import json, sys, gzip
sys.path.insert(0, "tools"); sys.path.insert(0, "experiments")
import numpy as np
from PIL import Image, ImageDraw
import change_ratio as cr
from e7_gate import load_geom
d = json.load(open("results/e9/change_ratio.json"))
sel = {c["clip"]: c for c in json.load(open("results/e7/selection.json"))["clips"]}
idx = {c["clip"]: c for c in json.load(open("data/meva/index.json"))["clips"]}
# a busy, active window near the median kept share (not the extreme)
cand = [(cl, r) for cl, rs in d["per_clip"].items() if sel[cl]["bin"] == "busy" for r in rs if r["active"] and r["keep"]["tracker"] > 0]
cand.sort(key=lambda x: x[1]["keep"]["pixel>2+dil"]); cl, r = cand[len(cand) // 2]
w = r["w"]; base = w * 60; fr = [base + o for o in cr.OFFSETS]
g = [cr.gray(cr.Path("data/meva/frames5") / cl / f"{f:05d}.jpg") for f in fr]
score = cr.block_mean(np.maximum(np.abs(g[6] - g[4]), np.abs(g[7] - g[5])))  # pair 3 vs pair 2
pm = cr.dilate(score > 2)
tracks = {json.loads(l)["f"]: [o[3:7] for o in json.loads(l)["o"]] for l in gzip.open(f"data/meva/tracks/{cl}.jsonl.gz", "rt") if l.strip()}
tm = cr.dilate(cr.box_mask(tracks.get(fr[6], []) + tracks.get(fr[7], [])))
geom = load_geom(cr.Path("data/meva/meva-data-repo") / idx[cl]["annotation_dir"] / f"{cl}.geom.yml")
im = Image.open(f"data/meva/frames5/{cl}/{fr[7]:05d}.jpg").convert("RGB").resize((896, 504))
ov = Image.new("RGBA", im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
for y in range(18):
    for x in range(32):
        bx = (x * 28, y * 28, x * 28 + 27, y * 28 + 27)
        if pm[y, x] and tm[y, x]: dr.rectangle(bx, fill=(160, 60, 200, 90))
        elif pm[y, x]: dr.rectangle(bx, fill=(220, 60, 20, 90))
        elif tm[y, x]: dr.rectangle(bx, fill=(30, 100, 220, 90))
for a in idx[cl]["activities"]:
    if a["start"] <= fr[7] <= a["end"]:
        for tid in a["actors"]:
            b = geom.get(tid, {}).get(fr[7])
            if b: dr.rectangle([v * 896 / 1920 if i % 2 == 0 else v * 504 / 1080 for i, v in enumerate(b)], outline=(255, 230, 0, 255), width=2)
out = Image.alpha_composite(im.convert("RGBA"), ov).convert("RGB")
ImageDraw.Draw(out).text((6, 6), f"{cl[-24:]} w{w}  red=pixel>2+dil  blue=tracker  purple=both  yellow=annotated actors", fill=(255, 255, 255))
out.save("results/e9/mask_example.png"); print(cl, w, r["keep"]["pixel>2+dil"], r["keep"]["tracker"])
