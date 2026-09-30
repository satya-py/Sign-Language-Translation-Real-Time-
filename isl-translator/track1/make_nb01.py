"""Builds 01_include_to_poses.ipynb (Kaggle notebook). Run: python make_nb01.py"""
import json

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                               "outputs": [], "source": s.strip("\n")})

md("""
# Track 1 · Step 1 — INCLUDE videos → pose files

Converts the INCLUDE ISL dataset (263 signs, 4,292 videos, 56.8 GB on Zenodo) into
MediaPipe Holistic `.pose` files, one zip at a time, so disk never fills up.

**Kaggle settings (right panel):** Accelerator = **None** (CPU is enough, saves GPU quota) ·
Internet = **On** · then *Save Version → Save & Run All (Commit)*.

Expected runtime: ~2–4 h for all categories. Output (`/kaggle/working/include_poses`) is
~2–3 GB. When it finishes: open the version's **Output** tab → **New Dataset** → name it
`include-poses` (private).
""")

code("""
!pip install -q "mediapipe==0.10.21" "pose-format==0.15.0"
""")

code("""
import os, re, csv, time, shutil, zipfile, requests, multiprocessing as mp
from pathlib import Path

# Which INCLUDE categories to convert. None = all 15 (57 GB, 3-4 h).
# Phase 1 for the live demo: greetings + pronouns + people + time (~15 GB, ~1.5 h).
CATEGORIES = ["Greetings", "Pronouns", "People", "Days_and_Time"]
# Phase 2 (run later if time allows): add "Adjectives", "Places", "Jobs"

OUT = Path("/kaggle/working/include_poses")
TMP = Path("/tmp/include_zip")
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)
WORKERS = os.cpu_count()
# Frames are downscaled to this width before pose estimation (faster, and closer to a
# laptop webcam's resolution than INCLUDE's 1920x1080).
TARGET_WIDTH = 960
print("workers:", WORKERS)
""")

code("""
# File list straight from the Zenodo record.
rec = requests.get("https://zenodo.org/api/records/4010759", timeout=60).json()
zips = sorted([f for f in rec["files"] if f["key"].endswith(".zip")], key=lambda f: f["key"])
if CATEGORIES:
    zips = [f for f in zips if f["key"].rsplit("_", 1)[0] in CATEGORIES]
print(len(zips), "zips,", round(sum(f["size"] for f in zips) / 1e9, 1), "GB")

# Official INCLUDE train/val/test split, so our Track 1 test clips are never trained on.
SPLIT = {}
for s in ["train", "val", "test"]:
    url = f"https://raw.githubusercontent.com/AI4Bharat/INCLUDE/master/train_test_paths/include_{s}.txt"
    for line in requests.get(url, timeout=60).text.splitlines():
        if line.strip():
            SPLIT[line.strip()] = s
print({s: list(SPLIT.values()).count(s) for s in ["train", "val", "test"]})
""")

code("""
def clean_label(folder):
    # "48. Hello" -> "hello", "Thank you" -> "thank_you" (folder name, lowercased)
    name = re.sub(r"^\s*\d+\.\s*", "", folder).strip().lower()
    return re.sub(r"[^a-z]+", "_", name).strip("_")


def video_to_pose(args):
    video, out_path = args
    if os.path.exists(out_path):
        return out_path, "exists", 0
    try:
        import cv2
        from simple_video_utils.metadata import video_metadata
        from simple_video_utils.frames import read_frames_exact
        from pose_format.utils.holistic import load_holistic
        meta = video_metadata(video)
        scale = min(1.0, TARGET_WIDTH / meta.width)
        w, h = int(meta.width * scale), int(meta.height * scale)
        frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) if scale < 1 else f
                  for f in read_frames_exact(video)]
        if not frames:
            return out_path, "no frames", 0
        pose = load_holistic(frames, fps=meta.fps, width=w, height=h, progress=False,
                             additional_holistic_config={"model_complexity": 1})
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            pose.write(f)
        return out_path, "ok", len(frames)
    except Exception as e:
        return out_path, f"error: {type(e).__name__}: {e}", 0
""")

code("""
rows, t_start = [], time.time()
for i, zf in enumerate(zips, 1):
    key, category = zf["key"], zf["key"].rsplit("_", 1)[0]
    zpath = TMP / key
    t0 = time.time()

    # Download (streamed) with a couple of retries.
    for attempt in range(3):
        try:
            with requests.get(zf["links"]["self"], stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(zpath, "wb") as f:
                    for chunk in r.iter_content(8 << 20):
                        f.write(chunk)
            if zpath.stat().st_size == zf["size"]:
                break
            print("  size mismatch, retrying", key)
        except Exception as e:
            print("  download failed, retrying:", key, e)
    else:
        print("GIVING UP on", key)
        continue

    extract_dir = TMP / key.replace(".zip", "")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(extract_dir)
    zpath.unlink()

    jobs = []
    for video in sorted(extract_dir.rglob("*")):
        if video.suffix.lower() not in (".mov", ".mp4"):
            continue
        rel = video.relative_to(extract_dir).as_posix()           # "Greetings/48. Hello/MVI_0029.MOV"
        label = clean_label(video.parent.name)
        out_path = OUT / label / (video.stem + ".pose")
        jobs.append((str(video), str(out_path), rel, label, category))

    with mp.Pool(WORKERS) as pool:
        results = pool.map(video_to_pose, [(j[0], j[1]) for j in jobs], chunksize=1)

    for (video, out_path, rel, label, category), (_, status, n) in zip(jobs, results):
        rows.append({"pose": os.path.relpath(out_path, OUT), "label": label, "category": category,
                     "split": SPLIT.get(rel, "unlisted"), "frames": n, "status": status, "source": rel})
    shutil.rmtree(extract_dir, ignore_errors=True)

    ok = sum(r[1] == "ok" for r in results)
    print(f"[{i}/{len(zips)}] {key}: {ok}/{len(jobs)} videos ok in {time.time()-t0:.0f}s "
          f"(total {(time.time()-t_start)/60:.0f} min)", flush=True)
""")

code("""
with open(OUT / "index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

import pandas as pd
df = pd.DataFrame(rows)
print(df.status.value_counts().head(10))
print(df[df.status == "ok"].split.value_counts())
print("labels:", df.label.nunique())
print(df[df.status != "ok"].head(20))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("01_include_to_poses.ipynb", "w"), indent=1)
print("wrote 01_include_to_poses.ipynb")
