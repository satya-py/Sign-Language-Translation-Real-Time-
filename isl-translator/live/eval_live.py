"""Honest end-to-end test: replay clips through the LIVE pipeline, not the trainer.

Each keypoint file is fed frame by frame to the same SignSpotter + classifier the
demo uses, so this measures what the audience will actually see: segmentation
mistakes, low-confidence rejections and all.

    python eval_live.py --model models/signs.pt
    python eval_live.py --model models/signs.pt --threshold 0.5
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from live_demo import Recogniser, SignSpotter  # noqa: E402
from train import load_clips, split_clips  # noqa: E402


def replay(rows, rec, threshold):
    """Returns the glosses the live app would emit for this clip."""
    spotter = SignSpotter()
    out = []
    for row in rows:
        segment = spotter.update(row)
        if segment is not None:
            label, prob = rec.predict(segment)[0]
            if label != "idle" and prob >= threshold:
                out.append((label, prob))
    if spotter.active and len(spotter.buffer) >= spotter.min_frames:
        label, prob = rec.predict(spotter.buffer)[0]     # clip ended mid-sign
        if label != "idle" and prob >= threshold:
            out.append((label, prob))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(HERE / "models" / "signs.pt"))
    p.add_argument("--keypoints", default=str(HERE / "data" / "keypoints"))
    p.add_argument("--recordings", default=str(HERE / "recordings"))
    p.add_argument("--threshold", type=float, default=0.55)
    p.add_argument("--val-only", action="store_true",
                   help="only clips held out during training (same seed)")
    args = p.parse_args()

    rec = Recogniser(args.model)
    clips = load_clips([args.keypoints, args.recordings])
    if args.val_only:
        _, clips = split_clips(clips)
    print(f"{len(clips)} clips, model classes: {rec.labels}")

    correct = missed = wrong = 0
    confusion = defaultdict(Counter)
    for c in clips:
        emitted = replay(c["rows"], rec, args.threshold)
        labels = [l for l, _ in emitted]
        if not labels:
            missed += 1
            confusion[c["label"]]["(nothing)"] += 1
        elif labels[0] == c["label"]:
            correct += 1
            confusion[c["label"]][labels[0]] += 1
        else:
            wrong += 1
            confusion[c["label"]][labels[0]] += 1

    total = len(clips)
    print(f"\nfirst sign correct : {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"nothing detected   : {missed} ({missed/total*100:.1f}%)")
    print(f"wrong sign         : {wrong} ({wrong/total*100:.1f}%)")
    print("\nper word:")
    for label in sorted(confusion):
        row = confusion[label]
        n = sum(row.values())
        top = ", ".join(f"{k} {v}" for k, v in row.most_common(3))
        print(f"  {label:<14} {row[label]}/{n}   [{top}]")

    report = {"model": args.model, "threshold": args.threshold, "clips": total,
              "correct": correct, "missed": missed, "wrong": wrong,
              "confusion": {k: dict(v) for k, v in confusion.items()}}
    out = Path(args.model).with_suffix(".live_eval.json")
    json.dump(report, open(out, "w"), indent=1)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
