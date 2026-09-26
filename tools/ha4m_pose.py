#!/usr/bin/env python3
"""R2: the RGB-only hand finder -- YOLOv8s-pose on every sampled HA4M frame.

The worker is the largest detected person (a second person's arm reaches in at the
frame edge in setup 1). Saved per frame: that person's box and COCO keypoints
5/6 (shoulders) and 9/10 (wrists), with confidences. No labels are used.

    docker/run_yolo.sh python3 tools/ha4m_pose.py        # -> data/ha4m/pose.json
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    from ultralytics import YOLO
    root = Path("data/ha4m")
    weights = Path("/models/yolov8s-pose.pt")
    model = YOLO(str(weights) if weights.exists() else "yolov8s-pose.pt")
    manifest = json.loads((root / "manifest.json").read_text())
    frames = sorted({(s["rec"], f) for s in manifest if s["complete"] for f in s["frames"]})
    out, missing = {}, 0
    batch = 16
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        paths = [str(root / r / "color" / f"{f:06d}.png") for r, f in chunk]
        for (rec, f), res in zip(chunk, model.predict(paths, imgsz=1280, conf=0.25, verbose=False)):
            if res.boxes is None or len(res.boxes) == 0:
                missing += 1
                continue
            areas = ((res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0]) *
                     (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1]))
            k = int(areas.argmax())
            kp = res.keypoints.data[k].tolist()          # 17 x (x, y, conf)
            out[f"{rec}|{f}"] = {"box": [round(v, 1) for v in res.boxes.xyxy[k].tolist()],
                                 "kp": {j: [round(kp[j][0], 1), round(kp[j][1], 1), round(kp[j][2], 3)]
                                        for j in (5, 6, 9, 10)}}
        if i % (batch * 50) == 0:
            print(f"  {i}/{len(frames)}", flush=True)
    (root / "pose.json").write_text(json.dumps(out))
    print(f"frames {len(frames)}, person found {len(out)}, none {missing}", flush=True)


if __name__ == "__main__":
    main()
