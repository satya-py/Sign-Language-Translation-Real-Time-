"""INCLUDE videos -> keypoint files, in the same format the live app produces.

Reads each video frame by frame (low RAM), runs MediaPipe Holistic, and saves
(frames, 300) float32 arrays: pose(33) + left hand(21) + right hand(21), each
(x, y, z, confidence). Identical layout to record.py, so training data and live
input always match.

    python extract_keypoints.py                    # all words under data/videos
    python extract_keypoints.py --workers 3
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

VIDEO_DIR = HERE / "data" / "videos"
KEYPOINT_DIR = HERE / "data" / "keypoints"
TARGET_WIDTH = 640  # close to a laptop webcam; also much faster than 1920x1080


def process_video(job):
    video, out_path = Path(job[0]), Path(job[1])
    if out_path.exists():
        return str(out_path), "exists", 0
    try:
        import cv2
        import mediapipe as mp
        from record import landmarks_to_row

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            return str(out_path), "cannot open", 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                                  min_tracking_confidence=0.5,
                                                  model_complexity=1)
        rows = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > TARGET_WIDTH:
                scale = TARGET_WIDTH / w
                frame = cv2.resize(frame, (TARGET_WIDTH, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            rows.append(landmarks_to_row(holistic.process(rgb)))
        cap.release()
        holistic.close()
        if len(rows) < 8:
            return str(out_path), f"too short ({len(rows)} frames)", len(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, keypoints=np.asarray(rows, dtype=np.float32),
                            label=out_path.parent.name, signer="include",
                            fps=float(fps), source=video.name)
        return str(out_path), "ok", len(rows)
    except Exception as e:
        return str(out_path), f"error: {type(e).__name__}: {e}", 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", default=str(VIDEO_DIR))
    p.add_argument("--out", default=str(KEYPOINT_DIR))
    p.add_argument("--workers", type=int, default=3,
                   help="keep at 3 or less: each worker holds its own MediaPipe models")
    args = p.parse_args()

    video_root, out_root = Path(args.videos), Path(args.out)
    jobs = []
    for video in sorted(video_root.rglob("*")):
        if video.suffix.lower() not in (".mov", ".mp4", ".avi"):
            continue
        jobs.append((str(video), str(out_root / video.parent.name / (video.stem + ".npz"))))
    if not jobs:
        sys.exit(f"no videos under {video_root}")
    todo = [j for j in jobs if not Path(j[1]).exists()]
    print(f"{len(jobs)} videos, {len(todo)} still to process", flush=True)

    t0, done, failures = time.time(), 0, []
    with Pool(args.workers) as pool:
        for path, status, frames in pool.imap_unordered(process_video, todo, chunksize=1):
            done += 1
            if status not in ("ok", "exists"):
                failures.append((path, status))
            if done % 20 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}  {(time.time()-t0)/60:.1f} min  "
                      f"({len(failures)} failed)", flush=True)

    counts = {d.name: len(list(d.glob('*.npz'))) for d in sorted(out_root.iterdir())
              if d.is_dir()}
    print("keypoint files per word:", counts)
    if failures:
        print("failures:", failures[:10])


if __name__ == "__main__":
    main()
