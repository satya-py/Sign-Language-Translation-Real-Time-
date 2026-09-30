"""Recorder for the live demo: webcam -> MediaPipe keypoints -> .npz per take.

Records the same keypoints the live app will use, from the same camera, with the
same code. That match is what makes the trained model work live.

    python record.py --signer satya                 # record the default word list
    python record.py --signer satya --words hello,i,you --takes 15
    python record.py --signer satya --idle           # record the "not signing" class

Keys:  SPACE = start/stop a take   N = next word   B = previous word
       D = delete last take        Q = quit (everything is saved as you go)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "recordings"

# Signs chosen so short sentences can be built. INCLUDE has no verbs, so these
# are the ones worth recording ourselves.
DEFAULT_WORDS = [
    "hello", "thank_you", "sorry", "yes", "no",
    "i", "you", "doctor", "water", "food",
    "want", "need", "help", "go", "come",
    "pain", "hospital", "toilet", "good", "bad",
]

N_POSE, N_HAND = 33, 21
FEATURE_DIM = (N_POSE + 2 * N_HAND) * 4  # x, y, z, confidence -> 300, like iSign


def landmarks_to_row(results):
    """One frame -> 300 numbers, in the iSign order: pose, left hand, right hand.
    Missing parts become NaN so they can be interpolated or masked later."""
    row = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    parts = [(results.pose_landmarks, N_POSE, 0),
             (results.left_hand_landmarks, N_HAND, N_POSE * 4),
             (results.right_hand_landmarks, N_HAND, (N_POSE + N_HAND) * 4)]
    for lms, count, offset in parts:
        if lms is None:
            continue
        for i, lm in enumerate(lms.landmark[:count]):
            conf = getattr(lm, "visibility", 1.0)
            row[offset + i * 4: offset + i * 4 + 4] = (lm.x, lm.y, lm.z, conf)
    return row


def draw(frame, word, idx, total, takes, recording, seconds, message):
    h, w = frame.shape[:2]
    bar = frame[:110].copy()
    cv2.rectangle(bar, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame[:110], 0.45, 0, frame[:110])

    cv2.putText(frame, word.upper().replace("_", " "), (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
    cv2.putText(frame, word.upper().replace("_", " "), (16, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 255, 120), 2)
    cv2.putText(frame, f"word {idx + 1}/{total}   takes saved: {takes}", (16, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
    cv2.putText(frame, "SPACE record | N next | B back | D delete last | Q quit",
                (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if message:
        cv2.putText(frame, message, (16, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 200, 255), 1)
    if recording:
        cv2.circle(frame, (w - 40, 40), 14, (60, 60, 255), -1)
        cv2.putText(frame, f"REC {seconds:.1f}s", (w - 165, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 255), 2)
    return frame


def save_take(frames, word, signer, fps):
    out_dir = DATA / word
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = out_dir / f"{signer}_{stamp}.npz"
    np.savez_compressed(path, keypoints=np.asarray(frames, dtype=np.float32),
                        label=word, signer=signer, fps=fps)
    return path


def count_takes(word):
    d = DATA / word
    return len(list(d.glob("*.npz"))) if d.exists() else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signer", required=True, help="who is signing (keeps takes separable)")
    p.add_argument("--words", default=None, help="comma separated; default is the built-in list")
    p.add_argument("--takes", type=int, default=15, help="target takes per word")
    p.add_argument("--idle", action="store_true",
                   help="record the 'not signing' class: sit, move, adjust hair, talk")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--auto", type=int, default=0,
                   help="testing only: record N takes of 2s automatically, then exit")
    args = p.parse_args()

    words = ["idle"] if args.idle else (args.words.split(",") if args.words else DEFAULT_WORDS)
    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5, model_complexity=1)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit(f"camera {args.camera} did not open")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window = "recorder"
    if not args.auto:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, args.width, args.height)

    idx, recording, frames, t_start, message, last_saved = 0, False, [], 0.0, "", None
    saved_total, auto_done = 0, 0
    fps_target = 25.0
    print(f"recording into {DATA}  (target {args.takes} takes per word)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                break
            word = words[idx]

            if recording:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                frames.append(landmarks_to_row(holistic.process(rgb)))

            if args.auto:
                if not recording:
                    recording, frames, t_start = True, [], time.time()
                elif time.time() - t_start > 2.0:
                    path = save_take(frames, word, args.signer, fps_target)
                    saved_total += 1
                    auto_done += 1
                    print(f"[auto] saved {path.name}: {len(frames)} frames")
                    recording = False
                    if auto_done >= args.auto:
                        break
                continue

            shown = draw(cv2.flip(frame, 1), word, idx, len(words), count_takes(word),
                         recording, time.time() - t_start if recording else 0.0, message)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not recording:
                    recording, frames, t_start, message = True, [], time.time(), ""
                else:
                    recording = False
                    if len(frames) >= 8:
                        last_saved = save_take(frames, word, args.signer, fps_target)
                        saved_total += 1
                        message = f"saved {len(frames)} frames"
                        if count_takes(word) >= args.takes and idx + 1 < len(words):
                            idx += 1
                            message += " -> next word"
                    else:
                        message = "too short, not saved"
            elif key in (ord("n"), ord("N")):
                idx, recording, message = (idx + 1) % len(words), False, ""
            elif key in (ord("b"), ord("B")):
                idx, recording, message = (idx - 1) % len(words), False, ""
            elif key in (ord("d"), ord("D")) and last_saved and last_saved.exists():
                last_saved.unlink()
                saved_total -= 1
                message = f"deleted {last_saved.name}"
                last_saved = None
            elif key in (ord("q"), 27):
                break
            elif not args.auto and cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        holistic.close()
        if not args.auto:
            cv2.destroyAllWindows()
        summary = {w: count_takes(w) for w in words}
        (DATA).mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(DATA / "summary.json", "w"), indent=1)
        print(f"saved {saved_total} takes this session")
        print("takes per word:", summary)


if __name__ == "__main__":
    main()
