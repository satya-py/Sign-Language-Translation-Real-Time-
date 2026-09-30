"""Record single-frame keypoints for ISL fingerspelling (A-Z, 0-10).

Unlike record.py (which records a moving clip per sign), fingerspelling letters
are STATIC hand poses. So this saves individual frames, not clips: each SPACE
press (or each auto-capture tick while holding) saves one (300,) keypoint row,
the exact same format landmarks_to_row produces in the live app.

    python record_letters.py --signer satya                       # full A-Z + 0-10
    python record_letters.py --signer satya --labels A,B,C,1,2
    python record_letters.py --signer satya --burst                # hold SPACE, auto-captures

Keys:  SPACE = capture one frame   H = hold-to-burst (10 frames over 2s)
       N = next letter   B = previous letter   D = delete last capture   Q = quit
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
LIVE_DIR = HERE.parent / "live"          # reuse the tested landmark extraction
sys.path.insert(0, str(LIVE_DIR))
from record import landmarks_to_row  # noqa: E402

DATA = HERE / "data" / "letters"

# A-Z and 0-10. Change with --labels if you only want a subset first
# (e.g. the one-handed, unambiguous letters: A,I,L,O,U,Y).
DEFAULT_LABELS = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + \
                 [str(n) for n in range(0, 11)]

BURST_FRAMES = 10
BURST_SECONDS = 2.0


def count_saved(label):
    d = DATA / label
    return len(list(d.glob("*.npz"))) if d.exists() else 0


def save_frame(row, label, signer):
    out_dir = DATA / label
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = out_dir / f"{signer}_{stamp}.npz"
    np.savez_compressed(path, keypoints=row.astype(np.float32), label=label,
                        signer=signer, mirrored=True)   # already in web orientation
    return path


def draw(frame, label, idx, total, saved, mode_msg, message):
    h, w = frame.shape[:2]
    bar = frame[:110].copy()
    cv2.rectangle(bar, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.55, frame[:110], 0.45, 0, frame[:110])

    cv2.putText(frame, label, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 7)
    cv2.putText(frame, label, (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (80, 255, 120), 3)
    cv2.putText(frame, f"letter {idx + 1}/{total}   saved: {saved}", (16, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
    cv2.putText(frame, "SPACE capture | H burst(2s) | N next | B back | D delete | Q quit",
                (16, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if mode_msg:
        cv2.putText(frame, mode_msg, (w - 260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (60, 60, 255), 2)
    if message:
        cv2.putText(frame, message, (16, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 200, 255), 1)
    return frame


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signer", required=True, help="who is signing (keeps captures separable)")
    p.add_argument("--labels", default=None,
                   help="comma separated; default is A-Z and 0-10")
    p.add_argument("--target", type=int, default=30, help="target captures per letter")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    args = p.parse_args()

    labels = args.labels.split(",") if args.labels else DEFAULT_LABELS

    import mediapipe as mp
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                              min_tracking_confidence=0.5, model_complexity=1)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened():
        sys.exit(f"camera {args.camera} did not open")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window = "fingerspelling recorder"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)

    idx, message, saved_total = 0, "", 0
    bursting, burst_start, burst_count = False, 0.0, 0
    next_burst_capture = 0.0

    print(f"labels: {labels}\ncaptures go to {DATA}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                break
            label = labels[idx]
            now = time.time()
            # Mirror only what the signer sees. MediaPipe must read the camera
            # image as it is, because that is what the web app sends it - running it
            # on the flipped preview trained every letter the wrong way round, and
            # fingerspelling failed live while scoring 96% in testing.
            flipped = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            row = landmarks_to_row(holistic.process(rgb))

            mode_msg = ""
            if bursting:
                mode_msg = f"BURST {burst_count}/{BURST_FRAMES}"
                if now >= next_burst_capture and burst_count < BURST_FRAMES:
                    save_frame(row, label, args.signer)
                    saved_total += 1
                    burst_count += 1
                    next_burst_capture = now + (BURST_SECONDS / BURST_FRAMES)
                if burst_count >= BURST_FRAMES or now - burst_start > BURST_SECONDS + 0.5:
                    bursting = False
                    message = f"burst done: +{burst_count} frames"

            shown = draw(flipped.copy(), label, idx, len(labels), count_saved(label),
                        mode_msg, message)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                save_frame(row, label, args.signer)
                saved_total += 1
                message = f"captured (total {count_saved(label)})"
                if count_saved(label) >= args.target and idx + 1 < len(labels):
                    idx += 1
                    message += " -> next letter"
            elif key in (ord("h"), ord("H")):
                bursting, burst_start, burst_count = True, now, 0
                next_burst_capture = now
                message = ""
            elif key in (ord("n"), ord("N")):
                idx, message = (idx + 1) % len(labels), ""
            elif key in (ord("b"), ord("B")):
                idx, message = (idx - 1) % len(labels), ""
            elif key in (ord("d"), ord("D")):
                d = DATA / label
                files = sorted(d.glob("*.npz")) if d.exists() else []
                if files:
                    files[-1].unlink()
                    saved_total -= 1
                    message = f"deleted {files[-1].name}"
            elif key in (ord("q"), 27):
                break
            elif cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()
        summary = {l: count_saved(l) for l in labels}
        DATA.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(DATA / "summary.json", "w"), indent=1)
        print(f"saved {saved_total} frames this session")
        print("captures per label:", summary)


if __name__ == "__main__":
    main()