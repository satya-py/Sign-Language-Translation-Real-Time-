"""Flip the recorded fingerspelling frames to match what the web app sends.

record_letters.py used to run MediaPipe on the mirrored preview (cv2.flip), while
the browser sends the camera image as it is. Every letter was therefore trained
with the hands the wrong way round, which is why fingerspelling worked in testing
and failed live - the model was being shown a mirror image of what it learned.

This rewrites each recorded frame once, as if it had been captured unmirrored:
x -> 1 - x, the left and right hand blocks swapped, and the left/right pose
landmarks swapped. It refuses to run twice by writing "mirrored" into the file.

    python mirror_dataset.py --dry-run
    python mirror_dataset.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "live"))

N_POSE, N_HAND = 33, 21
POSE_END = N_POSE * 4
LEFT_HAND = slice(POSE_END, POSE_END + N_HAND * 4)
RIGHT_HAND = slice(POSE_END + N_HAND * 4, POSE_END + 2 * N_HAND * 4)

# MediaPipe pose landmarks come in left/right pairs; mirroring swaps each pair.
POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
              (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
              (31, 32)]


def mirror_row(row):
    """The same frame as it would have looked without cv2.flip."""
    out = np.asarray(row, dtype=np.float32).copy()
    out[0::4] = 1.0 - out[0::4]                       # every x, including NaN handling
    left = out[LEFT_HAND].copy()
    out[LEFT_HAND] = out[RIGHT_HAND]
    out[RIGHT_HAND] = left
    for a, b in POSE_PAIRS:
        pa, pb = slice(a * 4, a * 4 + 4), slice(b * 4, b * 4 + 4)
        keep = out[pa].copy()
        out[pa] = out[pb]
        out[pb] = keep
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(HERE / "data" / "letters"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = Path(args.data)
    if not root.exists():
        sys.exit(f"no data at {root}")
    changed = already = 0
    for path in sorted(root.rglob("*.npz")):
        data = dict(np.load(path, allow_pickle=True))
        if bool(data.get("mirrored", False)):
            already += 1
            continue
        data["keypoints"] = mirror_row(data["keypoints"])
        data["mirrored"] = True
        if not args.dry_run:
            np.savez_compressed(path, **data)
        changed += 1
    what = "would flip" if args.dry_run else "flipped"
    print(f"{what} {changed} frames; {already} were already unmirrored")
    if changed and not args.dry_run:
        print("now retrain: python train_letters.py")


if __name__ == "__main__":
    main()
