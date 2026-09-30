"""Measure action.h5 on your own hands, letter by letter.

The model's accuracy inside the notebook is measured on frames from the same
sitting as the frames it trained on, which flatters it. This measures it the way
it will actually be used: you, your camera, now.

    python action_check.py                       # all 36, ten frames each
    python action_check.py --labels A,B,C --per 20
    python action_check.py --compare             # also score the PyTorch model

Show the letter on screen, hold it still, and press SPACE to record ten frames of
it. N skips, Q stops early. At the end it prints per-letter accuracy and the
confusions, and writes the numbers to action_check.json so they can go in a
README instead of being remembered.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from action_model import ACTIONS  # noqa: E402
from action_infer import ActionSpeller  # noqa: E402


def draw(frame, label, done, target, message, guesses):
    height, width = frame.shape[:2]
    bar = frame[:96].copy()
    cv2.rectangle(bar, (0, 0), (width, 96), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame[:96], 0.45, 0, frame[:96])
    cv2.putText(frame, f"SHOW: {label}", (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (80, 255, 120), 2)
    cv2.putText(frame, f"{done}/{target} frames    SPACE record   N next   Q quit",
                (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    for i, (name, prob) in enumerate(guesses[:3]):
        colour = (80, 255, 120) if name == label else (200, 200, 200)
        cv2.putText(frame, f"{name} {prob:.0%}", (16, 140 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    if message:
        cv2.putText(frame, message, (16, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=",".join(ACTIONS))
    parser.add_argument("--per", type=int, default=10, help="frames to judge per letter")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out", default=str(HERE / "action_check.json"))
    args = parser.parse_args()

    labels = [l for l in args.labels.split(",") if l]
    speller = ActionSpeller(args.model)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not capture.isOpened():
        sys.exit(f"camera {args.camera} did not open")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    results = collections.defaultdict(lambda: {"right": 0, "seen": 0})
    confusions = collections.Counter()
    index, message = 0, ""
    window = "action.h5 check"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while index < len(labels):
            label = labels[index]
            ok, frame = capture.read()
            if not ok:
                break
            guesses = speller.classify(frame, top_k=3)
            shown = draw(cv2.flip(frame, 1).copy(), label,
                         results[label]["seen"], args.per, message, guesses)
            cv2.imshow(window, shown)
            key = cv2.waitKey(10) & 0xFF

            if key == ord(" "):
                # Judge several frames in a row: one frame is luck either way.
                taken = 0
                while taken < args.per:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    guesses = speller.classify(frame, top_k=1)
                    if not guesses:
                        cv2.imshow(window, draw(cv2.flip(frame, 1).copy(), label,
                                                results[label]["seen"], args.per,
                                                "no hand seen", []))
                        cv2.waitKey(10)
                        continue
                    guess = guesses[0][0]
                    results[label]["seen"] += 1
                    taken += 1
                    if guess == label:
                        results[label]["right"] += 1
                    else:
                        confusions[f"{label}->{guess}"] += 1
                    cv2.imshow(window, draw(cv2.flip(frame, 1).copy(), label,
                                            results[label]["seen"], args.per,
                                            f"read {guess}", guesses))
                    cv2.waitKey(10)
                got = results[label]
                message = f"{label}: {got['right']}/{got['seen']} correct"
                index += 1
            elif key in (ord("n"), ord("N")):
                index += 1
                message = ""
            elif key in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        speller.close()

    judged = {l: v for l, v in results.items() if v["seen"]}
    if not judged:
        print("nothing was judged")
        return 1
    right = sum(v["right"] for v in judged.values())
    seen = sum(v["seen"] for v in judged.values())
    print(f"\naction.h5 on your camera: {right}/{seen} = {100 * right / seen:.1f}% "
          f"over {len(judged)} letters")
    print("\nper letter:")
    for label in labels:
        if label in judged:
            v = judged[label]
            print(f"  {label:<3} {v['right']}/{v['seen']}")
    if confusions:
        print("\nmost common confusions:", confusions.most_common(10))
    weak = [l for l, v in judged.items() if v["right"] / v["seen"] < 0.6]
    if weak:
        print(f"\nweak letters ({len(weak)}): {', '.join(weak)}")

    Path(args.out).write_text(json.dumps({
        "model": str(speller.model.path),
        "at": time.strftime("%Y-%m-%d %H:%M"),
        "overall": {"right": right, "frames": seen, "accuracy": round(right / seen, 4)},
        "per_letter": {l: judged[l] for l in judged},
        "confusions": confusions.most_common(20),
    }, indent=1), encoding="utf-8")
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
